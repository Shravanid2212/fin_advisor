"""Data ingestion, cleaning, and metric computation for Mutual Funds (PRD Phase 19).

Features:
1. Downloads historical daily NAV from MFAPI service (api.mfapi.in).
2. Saves raw untouched JSON responses in data/raw/funds/<scheme_code>.json.
3. Cleans NAVs: converts DD-MM-YYYY to YYYY-MM-DD, filters invalid/NA values, sorts chronologically.
4. Computes 1Y, 3Y, 5Y CAGR, volatility, max drawdown, Sharpe, Sortino, rolling 3Y consistency, and beta.
5. Persists cleaned NAV history and metrics into DuckDB tables `fund_nav` and `fund_metrics`.
6. Enforces Phase 19 selection rule: categories with fewer than 6 usable funds are marked unranked.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.config import load_config
from src.funds.metrics import (
    calculate_annualized_volatility,
    calculate_beta,
    calculate_cagr,
    calculate_max_drawdown,
    calculate_rolling_consistency,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/market.duckdb")
RAW_FUNDS_DIR = Path("data/raw/funds")
REFERENCE_FUNDS_CSV = Path("data/reference/funds.csv")


def load_curated_funds_reference(csv_path: Path | str = REFERENCE_FUNDS_CSV) -> list[dict[str, Any]]:
    """Load curated mutual funds metadata from reference CSV file."""
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Reference funds file not found at: {p}")

    funds: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader((line for line in f if not line.startswith("#")))
        for r in reader:
            if r.get("scheme_code"):
                funds.append({
                    "scheme_code": int(r["scheme_code"]),
                    "name": r["name"].strip(),
                    "fund_house": r["fund_house"].strip(),
                    "category": r["category"].strip(),
                    "plan": r.get("plan", "Direct-Growth").strip(),
                })
    return funds


def download_fund_raw(scheme_code: int, raw_dir: Path = RAW_FUNDS_DIR, timeout: int = 15) -> Path | None:
    """Download raw JSON for a mutual fund scheme from MFAPI and save untouched.

    Args:
        scheme_code: AMFI scheme code integer.
        raw_dir: Target directory for raw files.
        timeout: Network timeout in seconds.

    Returns:
        Path to saved raw file, or None if download failed.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file = raw_dir / f"{scheme_code}.json"

    url = f"https://api.mfapi.in/mf/{scheme_code}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            # Validate JSON content
            parsed = json.loads(data.decode("utf-8"))
            if parsed.get("status") == "SUCCESS" and "data" in parsed:
                out_file.write_bytes(data)
                return out_file
            else:
                logger.warning(f"Scheme {scheme_code} returned non-success payload: {parsed.get('status')}")
                return None
    except Exception as exc:
        logger.warning(f"Failed to download scheme {scheme_code} from MFAPI: {exc}")
        return None


def clean_fund_nav_data(raw_json_path: Path) -> pd.DataFrame:
    """Clean raw JSON NAV records into a standardized, sorted DataFrame.

    Converts 'DD-MM-YYYY' dates into ISO 'YYYY-MM-DD', filters 'N.A.' and non-positive NAVs,
    and removes duplicate dates keeping the latest.
    """
    with open(raw_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("meta", {})
    records = payload.get("data", [])
    if not records:
        return pd.DataFrame(columns=["date", "nav"])

    cleaned_rows = []
    for r in records:
        d_str = r.get("date")
        n_str = r.get("nav")
        if not d_str or not n_str or n_str in ("N.A.", "None", "nan"):
            continue
        try:
            val = float(n_str)
            if val <= 0:
                continue
            # Parse DD-MM-YYYY
            parts = d_str.strip().split("-")
            if len(parts) == 3:
                day, month, year = parts
                iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                cleaned_rows.append({"date": iso_date, "nav": val})
        except (ValueError, TypeError):
            continue

    if not cleaned_rows:
        return pd.DataFrame(columns=["date", "nav"])

    df = pd.DataFrame(cleaned_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date"]).sort_values(by="date", ascending=True).reset_index(drop=True)
    return df


def compute_fund_metrics_record(
    scheme_code: int,
    meta: dict[str, Any],
    clean_df: pd.DataFrame,
    nifty_df: pd.DataFrame | None = None,
    risk_free_rate: float = 0.065,
    trading_days: int = 252,
) -> dict[str, Any] | None:
    """Compute performance and risk analytics for a single cleaned fund."""
    if clean_df.empty or len(clean_df) < 50:
        return None

    clean_df = clean_df.sort_values(by="date", ascending=True).reset_index(drop=True)
    clean_df["simple_return"] = clean_df["nav"].pct_change()

    last_nav = float(clean_df["nav"].iloc[-1])
    last_date = str(clean_df["date"].iloc[-1])
    first_nav = float(clean_df["nav"].iloc[0])
    first_date = str(clean_df["date"].iloc[0])

    dt_start = datetime.date.fromisoformat(first_date)
    dt_end = datetime.date.fromisoformat(last_date)
    total_elapsed_years = max(0.1, (dt_end - dt_start).days / 365.25)

    # 1Y, 3Y, 5Y CAGR
    def get_cagr_for_years(years: float) -> float:
        cutoff_dt = dt_end - datetime.timedelta(days=int(years * 365.25))
        cutoff_str = cutoff_dt.isoformat()
        sub = clean_df[clean_df["date"] <= cutoff_str]
        if sub.empty:
            return 0.0
        start_p = float(sub["nav"].iloc[-1])
        return calculate_cagr(start_p, last_nav, years)

    cagr_1y = get_cagr_for_years(1.0)
    cagr_3y = get_cagr_for_years(3.0) if total_elapsed_years >= 2.5 else 0.0
    cagr_5y = get_cagr_for_years(5.0) if total_elapsed_years >= 4.5 else 0.0

    # Volatility, Drawdown, Sharpe, Sortino
    daily_rets = clean_df["simple_return"].dropna()
    ann_vol = calculate_annualized_volatility(daily_rets, trading_days=trading_days)
    max_dd = calculate_max_drawdown(clean_df["nav"])

    # For Sharpe and Sortino, use 3Y CAGR if available, else 1Y or total CAGR
    effective_cagr = cagr_3y if cagr_3y != 0.0 else (cagr_1y if cagr_1y != 0.0 else calculate_cagr(first_nav, last_nav, total_elapsed_years))
    sharpe = calculate_sharpe_ratio(effective_cagr, ann_vol, risk_free_rate=risk_free_rate)
    sortino = calculate_sortino_ratio(daily_rets, effective_cagr, risk_free_rate=risk_free_rate, trading_days=trading_days)

    # Rolling 3-year consistency
    bench_nav_series = None
    bench_rets = None
    if nifty_df is not None and not nifty_df.empty:
        merged = pd.merge(clean_df[["date", "nav", "simple_return"]], nifty_df[["date", "close", "simple_return"]], on="date", suffixes=("_fund", "_bench"), how="inner")
        if not merged.empty:
            bench_nav_series = merged["close"]
            bench_rets = merged["simple_return_bench"]
            fund_rets_aligned = merged["simple_return_fund"]
            beta = calculate_beta(fund_rets_aligned, bench_rets)
        else:
            beta = 1.0
    else:
        beta = 1.0

    consistency = calculate_rolling_consistency(clean_df["nav"], benchmark_nav=bench_nav_series)

    return {
        "scheme_code": scheme_code,
        "name": meta["name"],
        "fund_house": meta["fund_house"],
        "category": meta["category"],
        "plan": meta.get("plan", "Direct-Growth"),
        "cagr_1y": round(float(cagr_1y), 4),
        "cagr_3y": round(float(cagr_3y), 4),
        "cagr_5y": round(float(cagr_5y), 4),
        "annual_volatility": round(float(ann_vol), 4),
        "max_drawdown": round(float(max_dd), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "consistency_score": round(float(consistency), 4),
        "beta": round(float(beta), 4),
        "last_nav": round(float(last_nav), 4),
        "last_nav_date": last_date,
        "history_years": round(float(total_elapsed_years), 2),
    }


def run_mutual_funds_pipeline(
    db_path: Path | str = DEFAULT_DB_PATH,
    reference_csv: Path | str = REFERENCE_FUNDS_CSV,
    raw_dir: Path = RAW_FUNDS_DIR,
    max_download_workers: int = 1,
) -> dict[str, Any]:
    """Execute the end-to-end mutual fund ingestion and analytics pipeline.

    1. Ingests raw JSON with resume-checkpoints.
    2. Cleans NAV series.
    3. Computes comprehensive risk and return metrics.
    4. Evaluates minimum 6 usable funds rule per category.
    5. Saves into DuckDB and clean Parquet files.
    """
    cfg = load_config()
    rf = cfg.risk_free_rate
    td = cfg.trading_days

    db_p = Path(db_path)
    curated_funds = load_curated_funds_reference(reference_csv)
    print(f"Loaded {len(curated_funds)} curated fund specifications from {reference_csv}")

    # 1. Download or use cached raw JSON
    checkpoint_file = raw_dir / "_checkpoint.json"
    downloaded_codes: set[int] = set()
    if checkpoint_file.exists():
        try:
            downloaded_codes = set(json.loads(checkpoint_file.read_text(encoding="utf-8")))
        except Exception:
            pass

    success_downloads = 0
    failed_downloads = 0

    print("Checking / downloading mutual fund NAV histories from MFAPI...")
    for idx, f_meta in enumerate(curated_funds):
        code = f_meta["scheme_code"]
        raw_f = raw_dir / f"{code}.json"

        if raw_f.exists() and code in downloaded_codes:
            continue

        res = download_fund_raw(code, raw_dir=raw_dir)
        if res is not None:
            downloaded_codes.add(code)
            success_downloads += 1
        else:
            failed_downloads += 1

        time.sleep(0.04)  # Politeness delay

    checkpoint_file.write_text(json.dumps(list(downloaded_codes)), encoding="utf-8")

    # 2. Load Nifty 50 benchmark prices for beta & consistency comparisons
    nifty_df = None
    if db_p.exists():
        try:
            con = duckdb.connect(str(db_p), read_only=True)
            nifty_df = con.execute("SELECT date, close, simple_return FROM prices WHERE ticker = '^NSEI' ORDER BY date ASC").fetchdf()
            con.close()
        except Exception as e:
            logger.warning(f"Could not load ^NSEI from DuckDB: {e}")

    # 3. Clean NAV data and compute metrics
    cleaned_nav_rows: list[dict[str, Any]] = []
    fund_metrics_list: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}

    for f_meta in curated_funds:
        code = f_meta["scheme_code"]
        raw_f = raw_dir / f"{code}.json"
        if not raw_f.exists():
            continue

        clean_df = clean_fund_nav_data(raw_f)
        if clean_df.empty or len(clean_df) < 50:
            continue

        # Prepare nav rows
        for _, row in clean_df.iterrows():
            cleaned_nav_rows.append({
                "scheme_code": code,
                "date": row["date"],
                "nav": float(row["nav"]),
            })

        m = compute_fund_metrics_record(
            scheme_code=code,
            meta=f_meta,
            clean_df=clean_df,
            nifty_df=nifty_df,
            risk_free_rate=rf,
            trading_days=td,
        )
        if m is not None:
            fund_metrics_list.append(m)
            cat = m["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    # 4. Mark is_ranked (categories with < 6 usable funds are not ranked)
    for m in fund_metrics_list:
        cat = m["category"]
        usable_in_cat = category_counts.get(cat, 0)
        m["is_ranked"] = bool(usable_in_cat >= 6 and m["history_years"] >= 3.0)

    # 5. Persist to DuckDB
    con = duckdb.connect(str(db_p), read_only=False)
    try:
        # Create fund_nav
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_nav (
                scheme_code BIGINT,
                date DATE,
                nav DOUBLE,
                PRIMARY KEY (scheme_code, date)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_metrics (
                scheme_code BIGINT PRIMARY KEY,
                name VARCHAR,
                fund_house VARCHAR,
                category VARCHAR,
                plan VARCHAR,
                cagr_1y DOUBLE,
                cagr_3y DOUBLE,
                cagr_5y DOUBLE,
                annual_volatility DOUBLE,
                max_drawdown DOUBLE,
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                consistency_score DOUBLE,
                beta DOUBLE,
                last_nav DOUBLE,
                last_nav_date DATE,
                history_years DOUBLE,
                is_ranked BOOLEAN
            )
            """
        )

        # Upsert fund_nav
        if cleaned_nav_rows:
            nav_df = pd.DataFrame(cleaned_nav_rows)
            con.execute("DELETE FROM fund_nav")
            con.register("df_nav", nav_df)
            con.execute("INSERT INTO fund_nav SELECT scheme_code, CAST(date AS DATE), nav FROM df_nav")
            con.unregister("df_nav")

        # Upsert fund_metrics
        if fund_metrics_list:
            metrics_df = pd.DataFrame(fund_metrics_list)
            con.execute("DELETE FROM fund_metrics")
            con.register("df_met", metrics_df)
            con.execute(
                """
                INSERT INTO fund_metrics
                SELECT
                    scheme_code, name, fund_house, category, plan,
                    cagr_1y, cagr_3y, cagr_5y, annual_volatility, max_drawdown,
                    sharpe_ratio, sortino_ratio, consistency_score, beta,
                    last_nav, CAST(last_nav_date AS DATE), history_years, is_ranked
                FROM df_met
                """
            )
            con.unregister("df_met")
    finally:
        con.close()

    # Also save parquet backups in data/clean and data/features
    if cleaned_nav_rows:
        Path("data/clean").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cleaned_nav_rows).to_parquet("data/clean/fund_nav.parquet", index=False)
    if fund_metrics_list:
        Path("data/features").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fund_metrics_list).to_parquet("data/features/fund_metrics.parquet", index=False)

    return {
        "total_funds_processed": len(fund_metrics_list),
        "category_counts": category_counts,
        "unranked_categories": [cat for cat, count in category_counts.items() if count < 6],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_mutual_funds_pipeline()
    print("Mutual Funds Ingestion Completed:", res)
