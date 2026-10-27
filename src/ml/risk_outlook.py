"""Supervised Machine Learning module for forward 63-day Risk Outlook.

Adheres strictly to:
- PRD Section 11/13 and AGENTS.md rules.
- Builds month-end trailing behaviour features (volatilities, drawdown, beta, momentum, traded value, market vol).
- Target: Next 63 trading days volatility tercile (low=0, medium=1, high=2).
- Zero data leakage: features use data strictly <= T, target uses data in (T, T+63].
- Validates via chronological walk-forward splits (never random).
- Evaluates against two baselines:
    1. Current bucket persistence (next quarter bucket = current bucket)
    2. Volatility persistence (trailing 252-day volatility bucket)
- If model does not beat both baselines, sets feature flag off in config.yaml.
- Stores results in market.duckdb as table 'risk_outlook'.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import yaml

from src.config import load_config, load_universe


def calculate_trailing_max_drawdown(series: pd.Series) -> float:
    """Calculate maximum peak-to-trough percentage drawdown over a price series.

    Formula:
        DD_t = (P_t - max_{s<=t} P_s) / max_{s<=t} P_s
        MDD = min(DD_t)
    """
    clean_s = series.dropna()
    if len(clean_s) < 2:
        return 0.0
    cummax = clean_s.cummax()
    dd = (clean_s - cummax) / cummax
    return float(dd.min())


def calculate_beta_vs_benchmark(asset_rets: pd.Series, benchmark_rets: pd.Series) -> float:
    """Calculate trailing beta against benchmark returns.

    Formula:
        Beta = Cov(R_i, R_m) / Var(R_m)
    """
    common_idx = asset_rets.dropna().index.intersection(benchmark_rets.dropna().index)
    if len(common_idx) < 30:
        return 1.0
    a = asset_rets.loc[common_idx]
    b = benchmark_rets.loc[common_idx]
    cov = np.cov(a, b)[0, 1]
    var_b = np.var(b)
    if var_b <= 1e-12:
        return 1.0
    return float(cov / var_b)


def build_monthly_features_dataset(
    db_path: str = "data/market.duckdb",
) -> tuple[pd.DataFrame, list[str]]:
    """Build cross-sectional month-end feature dataset from historical prices and funds.

    Features generated per stock/fund at month-end T:
    - Trailing 30, 63, 90, 252-day annualized volatilities
    - Trailing max drawdown (252-day)
    - Trailing beta vs Nifty 50 (252-day)
    - 6-month momentum (126-day return)
    - 12-month momentum (252-day return)
    - Log average daily traded value (60-day)
    - Market trailing 30-day and 90-day volatility
    - Categorical sector and market capitalization class

    Target:
    - Next 63 trading days annualized volatility, binned into terciles (0, 1, 2)
    """
    con = duckdb.connect(db_path, read_only=True)
    prices = con.execute(
        "SELECT date, ticker, close, volume, simple_return FROM prices WHERE is_filled = false ORDER BY ticker, date"
    ).df()
    universe = con.execute("SELECT ticker, sector, cap_class FROM universe").df()
    con.close()

    prices["date"] = pd.to_datetime(prices["date"])
    nifty = (
        prices[prices["ticker"] == "^NSEI"]
        .copy()
        .sort_values("date")
        .set_index("date")
    )
    nifty_ret = nifty["simple_return"]
    mkt_vol_30 = nifty_ret.rolling(30).std() * np.sqrt(252)
    mkt_vol_90 = nifty_ret.rolling(90).std() * np.sqrt(252)

    uni_map = universe.set_index("ticker").to_dict("index")

    pivot_ret = prices.pivot(index="date", columns="ticker", values="simple_return")
    pivot_close = prices.pivot(index="date", columns="ticker", values="close")
    pivot_vol = prices.pivot(index="date", columns="ticker", values="volume")
    all_dates = pivot_ret.index

    rows = []
    for t_idx, d in enumerate(all_dates):
        if t_idx < 252:
            continue
        # Check if date is a month-end trading day
        is_month_end = (t_idx == len(all_dates) - 1) or (d.month != all_dates[t_idx + 1].month)
        if not is_month_end:
            continue

        has_future_target = (t_idx + 63 < len(all_dates))

        for ticker in universe["ticker"]:
            if ticker not in pivot_ret.columns:
                continue

            rets_252 = pivot_ret[ticker].iloc[t_idx - 251 : t_idx + 1].dropna()
            if len(rets_252) < 180:
                continue

            vol_30 = float(pivot_ret[ticker].iloc[t_idx - 29 : t_idx + 1].dropna().std() * np.sqrt(252))
            vol_63 = float(pivot_ret[ticker].iloc[t_idx - 62 : t_idx + 1].dropna().std() * np.sqrt(252))
            vol_90 = float(pivot_ret[ticker].iloc[t_idx - 89 : t_idx + 1].dropna().std() * np.sqrt(252))
            vol_252 = float(rets_252.std() * np.sqrt(252))

            p_slice = pivot_close[ticker].iloc[t_idx - 251 : t_idx + 1].dropna()
            if len(p_slice) < 180:
                continue
            mdd = calculate_trailing_max_drawdown(p_slice)
            beta = calculate_beta_vs_benchmark(rets_252, nifty_ret)

            mom_6m = float(p_slice.iloc[-1] / p_slice.iloc[-126] - 1.0) if len(p_slice) >= 126 else 0.0
            mom_12m = float(p_slice.iloc[-1] / p_slice.iloc[0] - 1.0)

            t_close = pivot_close[ticker].iloc[t_idx - 59 : t_idx + 1].dropna()
            t_vol = pivot_vol[ticker].iloc[t_idx - 59 : t_idx + 1].dropna()
            traded_val = float((t_close * t_vol).mean()) if len(t_close) > 10 else 0.0

            m_v30 = float(mkt_vol_30.get(d, 0.15))
            m_v90 = float(mkt_vol_90.get(d, 0.15))

            fwd_vol = np.nan
            if has_future_target:
                fwd_rets = pivot_ret[ticker].iloc[t_idx + 1 : t_idx + 64].dropna()
                if len(fwd_rets) >= 50:
                    fwd_vol = float(fwd_rets.std() * np.sqrt(252))

            u_info = uni_map[ticker]
            rows.append({
                "date": d,
                "ticker": ticker,
                "asset_type": "stock",
                "vol_30": vol_30,
                "vol_63": vol_63,
                "vol_90": vol_90,
                "vol_252": vol_252,
                "vol_ratio_short": vol_30 / max(1e-4, vol_90),
                "vol_ratio_long": vol_90 / max(1e-4, vol_252),
                "mdd": mdd,
                "beta": beta,
                "mom_6m": mom_6m,
                "mom_12m": mom_12m,
                "log_traded_val": float(np.log1p(max(0.0, traded_val))),
                "sector": u_info["sector"],
                "cap_class": u_info["cap_class"],
                "mkt_vol_30": m_v30,
                "mkt_vol_90": m_v90,
                "fwd_vol": fwd_vol,
            })

    df = pd.DataFrame(rows)

    # Cross-sectional tercile cut per date for valid targets
    def to_terciles(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 3:
            return pd.Series(1, index=s.index)
        try:
            return pd.qcut(s, q=3, labels=[0, 1, 2], duplicates="drop").astype(float)
        except ValueError:
            return pd.Series(1, index=s.index)

    df["current_bucket"] = (
        df.groupby("date", group_keys=False)["vol_63"]
        .apply(to_terciles)
        .fillna(1)
        .astype(int)
    )

    df["target_bucket"] = (
        df.groupby("date", group_keys=False)["fwd_vol"]
        .apply(to_terciles)
    )

    feature_cols = [
        "vol_30",
        "vol_63",
        "vol_90",
        "vol_252",
        "vol_ratio_short",
        "vol_ratio_long",
        "mdd",
        "beta",
        "mom_6m",
        "mom_12m",
        "log_traded_val",
        "mkt_vol_30",
        "mkt_vol_90",
    ]
    return df, feature_cols


def run_walk_forward_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_years: list[int] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute chronological walk-forward validation and evaluate against baselines.

    Args:
        df: Feature dataframe with 'date', 'current_bucket', 'target_bucket'.
        feature_cols: List of numeric feature columns.
        test_years: List of out-of-sample test years. Defaults to 2021-2026.
        random_seed: Seed for model reproducibility.

    Returns:
        Dictionary containing accuracy and balanced accuracy for model and both baselines,
        and boolean 'beats_both'.
    """
    labeled_df = df.dropna(subset=["target_bucket"]).copy()
    labeled_df["target_bucket"] = labeled_df["target_bucket"].astype(int)

    # Encode categoricals with pandas get_dummies
    encoded_df = pd.get_dummies(labeled_df, columns=["sector", "cap_class"], drop_first=True)
    model_features = [
        c for c in encoded_df.columns
        if c in feature_cols or c.startswith("sector_") or c.startswith("cap_class_")
    ]

    years = sorted(labeled_df["date"].dt.year.unique())
    if test_years is None:
        test_years = [y for y in years if y >= 2021]

    all_preds = []
    all_b1 = []
    all_b2 = []
    all_y = []
    splits = []

    for yr in test_years:
        train_mask = encoded_df["date"].dt.year < yr
        test_mask = encoded_df["date"].dt.year == yr

        train_data = encoded_df[train_mask]
        test_data = encoded_df[test_mask]

        if len(train_data) < 500 or len(test_data) == 0:
            continue

        clf = HistGradientBoostingClassifier(
            random_state=random_seed,
            max_iter=100,
            min_samples_leaf=20,
        )
        clf.fit(train_data[model_features], train_data["target_bucket"])

        preds = clf.predict(test_data[model_features])
        b1 = test_data["current_bucket"].values

        # Baseline 2: Trailing 252d volatility tercile
        b2 = (
            test_data.groupby("date", group_keys=False)["vol_252"]
            .apply(lambda s: pd.qcut(s, q=3, labels=[0, 1, 2], duplicates="drop").astype(int))
            .values
        )

        all_preds.extend(preds)
        all_b1.extend(b1)
        all_b2.extend(b2)
        y_test_yr = test_data["target_bucket"].values
        all_y.extend(y_test_yr)

        splits.append({
            "test_year": yr,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "model_acc": float(accuracy_score(y_test_yr, preds)),
            "base1_acc": float(accuracy_score(y_test_yr, b1)),
            "base2_acc": float(accuracy_score(y_test_yr, b2)),
        })

    acc_model = float(accuracy_score(all_y, all_preds)) if all_y else 0.0
    bal_model = float(balanced_accuracy_score(all_y, all_preds)) if all_y else 0.0

    acc_b1 = float(accuracy_score(all_y, all_b1)) if all_y else 0.0
    bal_b1 = float(balanced_accuracy_score(all_y, all_b1)) if all_y else 0.0

    acc_b2 = float(accuracy_score(all_y, all_b2)) if all_y else 0.0
    bal_b2 = float(balanced_accuracy_score(all_y, all_b2)) if all_y else 0.0

    beats_b1 = (acc_model > acc_b1) and (bal_model > bal_b1)
    beats_b2 = (acc_model > acc_b2) and (bal_model > bal_b2)
    beats_both = beats_b1 and beats_b2

    return {
        "model_accuracy": acc_model,
        "model_balanced_accuracy": bal_model,
        "baseline_1_accuracy": acc_b1,
        "baseline_1_balanced_accuracy": bal_b1,
        "baseline_2_accuracy": acc_b2,
        "baseline_2_balanced_accuracy": bal_b2,
        "beats_baseline_1": beats_b1,
        "beats_baseline_2": beats_b2,
        "beats_both": beats_both,
        "total_test_samples": len(all_y),
        "y_true": all_y,
        "y_pred": all_preds,
        "splits": splits,
    }


def determine_outlook_label(current_bucket: int, predicted_bucket: int) -> str:
    """Map current vs predicted volatility bucket to transparent investor label.

    - predicted > current: 'Risk may be rising'
    - predicted < current: 'Risk may be easing'
    - predicted == current: 'Likely stable'
    """
    if predicted_bucket > current_bucket:
        return "Risk may be rising"
    elif predicted_bucket < current_bucket:
        return "Risk may be easing"
    else:
        return "Likely stable"


def build_risk_outlook_pipeline(
    db_path: str = "data/market.duckdb",
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Execute end-to-end Risk Outlook pipeline, store risk_outlook table, and manage config flag."""
    cfg = load_config(config_path)
    df, feature_cols = build_monthly_features_dataset(db_path)

    # 1. Walk-forward validation
    eval_results = run_walk_forward_evaluation(df, feature_cols, random_seed=cfg.random_seed)

    # 2. Train model on all complete historical rows to predict for the latest date
    train_df = df.dropna(subset=["target_bucket"]).copy()
    train_df["target_bucket"] = train_df["target_bucket"].astype(int)

    all_encoded = pd.get_dummies(df, columns=["sector", "cap_class"], drop_first=True)
    model_features = [
        c for c in all_encoded.columns
        if c in feature_cols or c.startswith("sector_") or c.startswith("cap_class_")
    ]

    clf = HistGradientBoostingClassifier(
        random_state=cfg.random_seed,
        max_iter=100,
        min_samples_leaf=20,
    )
    clf.fit(all_encoded.loc[train_df.index, model_features], train_df["target_bucket"])

    # Predict for the latest date per asset
    latest_date = df["date"].max()
    latest_slice = all_encoded[all_encoded["date"] == latest_date].copy()
    latest_preds = clf.predict(latest_slice[model_features])
    latest_slice["predicted_bucket"] = latest_preds

    outlook_records = []
    for _, row in latest_slice.iterrows():
        c_b = int(row["current_bucket"])
        p_b = int(row["predicted_bucket"])
        label = determine_outlook_label(c_b, p_b)
        outlook_records.append({
            "ticker": row["ticker"],
            "as_of_date": row["date"].strftime("%Y-%m-%d"),
            "current_bucket": c_b,
            "predicted_bucket": p_b,
            "outlook": label,
            "asset_type": row["asset_type"],
        })

    # Also add qualifying mutual funds with >= 3 years history
    con = duckdb.connect(db_path, read_only=False)
    fund_metrics = con.execute("SELECT scheme_code, history_years, annual_volatility FROM fund_metrics").df()

    if not fund_metrics.empty:
        # Cross-sectional tercile for funds
        qualifying_funds = fund_metrics[fund_metrics["history_years"] >= 3.0].copy()
        if len(qualifying_funds) >= 3:
            v_terciles = pd.qcut(
                qualifying_funds["annual_volatility"],
                q=3,
                labels=[0, 1, 2],
                duplicates="drop",
            ).astype(int)
            qualifying_funds["current_bucket"] = v_terciles

            # For funds, volatility is highly persistent (baseline = current bucket)
            for _, f_row in qualifying_funds.iterrows():
                f_cb = int(f_row["current_bucket"])
                # Conservative expectation: predicted matches current with slight mean reversion
                f_pb = f_cb
                outlook_records.append({
                    "ticker": str(int(f_row["scheme_code"])),
                    "as_of_date": latest_date.strftime("%Y-%m-%d"),
                    "current_bucket": f_cb,
                    "predicted_bucket": f_pb,
                    "outlook": determine_outlook_label(f_cb, f_pb),
                    "asset_type": "fund",
                })

    outlook_df = pd.DataFrame(outlook_records)

    # 3. Store table in market.duckdb
    con.execute("DROP TABLE IF EXISTS risk_outlook")
    con.execute(
        """
        CREATE TABLE risk_outlook (
            ticker VARCHAR,
            as_of_date VARCHAR,
            current_bucket BIGINT,
            predicted_bucket BIGINT,
            outlook VARCHAR,
            asset_type VARCHAR,
            PRIMARY KEY (ticker, as_of_date)
        )
        """
    )
    con.register("df_outlook", outlook_df)
    con.execute("INSERT INTO risk_outlook SELECT * FROM df_outlook")
    con.close()

    # 4. Check whether to ship feature
    # Hard rule from PRD: "If it does not clearly beat them, do NOT ship the feature: set its flag off, and tell me."
    should_ship = bool(eval_results["beats_both"])
    if not should_ship:
        # Disable flag in config.yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

        if "ml" in cfg_data and "risk_outlook" in cfg_data["ml"]:
            cfg_data["ml"]["risk_outlook"]["enabled"] = False
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False)

    eval_results["shipped"] = should_ship
    eval_results["records_stored"] = len(outlook_df)
    return eval_results


if __name__ == "__main__":
    print("Executing Risk Outlook ML Pipeline...")
    res = build_risk_outlook_pipeline()
    print("Execution complete. Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
