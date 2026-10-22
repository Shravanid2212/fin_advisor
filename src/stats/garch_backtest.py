"""GARCH(1,1) vs Rolling Volatility Baseline Backtest Engine (PRD §10).

Compares 5-day out-of-sample volatility forecasts from GARCH(1,1) against a simple
30-day historical rolling standard deviation baseline across multiple assets and evaluation dates.
"""

from __future__ import annotations

import logging
from typing import Any
from arch import arch_model
import numpy as np
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


def run_garch_vs_baseline_backtest(
    db_path: str = "data/market.duckdb",
    test_tickers: list[str] | None = None,
    eval_dates: list[str] | None = None,
    lookback_window: int = 500,
    forecast_horizon: int = 5,
    baseline_window: int = 30,
    trading_days: int = 252,
) -> dict[str, Any]:
    """Execute walk-forward backtest comparing GARCH(1,1) vs simple rolling volatility baseline.

    Args:
        db_path: Path to DuckDB database.
        test_tickers: List of tickers to evaluate (default: representative multi-cap basket).
        eval_dates: List of evaluation cutoff dates.
        lookback_window: Historical window to train GARCH (default 500 days).
        forecast_horizon: Days ahead to evaluate (default 5 days).
        baseline_window: Lookback days for simple rolling standard deviation (default 30).
        trading_days: Annualization trading days (default 252).

    Returns:
        Dictionary containing overall RMSE for GARCH and Baseline, per-ticker metrics, and detailed dataframe.
    """
    if test_tickers is None:
        test_tickers = ["^NSEI", "TCS.NS", "HDFCBANK.NS", "POLYCAB.NS", "GRANULES.NS"]

    if eval_dates is None:
        eval_dates = [
            "2023-03-31", "2023-06-30", "2023-09-29", "2023-12-29",
            "2024-03-28", "2024-06-28", "2024-09-30", "2024-12-31"
        ]

    con = duckdb.connect(db_path, read_only=True)
    tickers_str = ", ".join([f"'{t}'" for t in test_tickers])
    df_prices = con.execute(f"""
        SELECT ticker, date, simple_return
        FROM prices
        WHERE ticker IN ({tickers_str})
          AND is_filled = FALSE
          AND simple_return IS NOT NULL
        ORDER BY ticker, date
    """).fetchdf()
    con.close()

    eval_results = []

    for ticker in test_tickers:
        sub = df_prices[df_prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        dates = sub["date"].values

        for eval_d in eval_dates:
            past_idx = np.where(dates <= eval_d)[0]
            if len(past_idx) < 250:
                continue
            t_idx = past_idx[-1]

            if t_idx + forecast_horizon >= len(sub):
                continue

            train_rets = sub.iloc[: t_idx + 1]["simple_return"].tail(lookback_window).values
            future_rets = sub.iloc[t_idx + 1 : t_idx + 1 + forecast_horizon]["simple_return"].values

            # Realized future annualized volatility over horizon
            realized_vol = float(np.std(future_rets, ddof=1) * np.sqrt(trading_days))

            # 1. Baseline: historical rolling annualized standard deviation
            baseline_slice = train_rets[-baseline_window:]
            baseline_vol = float(np.std(baseline_slice, ddof=1) * np.sqrt(trading_days))

            # 2. GARCH(1,1) model
            try:
                am = arch_model(train_rets * 100.0, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off")
                fc = res.forecast(horizon=forecast_horizon)
                daily_var_pct = fc.variance.iloc[-1].values
                daily_vol_dec = np.sqrt(daily_var_pct) / 100.0
                garch_vol = float(np.mean(daily_vol_dec) * np.sqrt(trading_days))
            except Exception as exc:
                logger.warning("GARCH fit error for %s on %s: %s", ticker, eval_d, exc)
                garch_vol = baseline_vol

            eval_results.append({
                "ticker": ticker,
                "eval_date": eval_d,
                "realized_vol": round(realized_vol, 6),
                "baseline_vol": round(baseline_vol, 6),
                "garch_vol": round(garch_vol, 6),
                "baseline_err_sq": (baseline_vol - realized_vol) ** 2,
                "garch_err_sq": (garch_vol - realized_vol) ** 2,
            })

    df_eval = pd.DataFrame(eval_results)
    if df_eval.empty:
        return {"error": "No evaluations could be completed."}

    rmse_baseline = float(np.sqrt(df_eval["baseline_err_sq"].mean()))
    rmse_garch = float(np.sqrt(df_eval["garch_err_sq"].mean()))

    ticker_summary = df_eval.groupby("ticker").agg(
        evaluations=("realized_vol", "count"),
        mean_realized_vol=("realized_vol", "mean"),
        rmse_baseline=("baseline_err_sq", lambda s: float(np.sqrt(s.mean()))),
        rmse_garch=("garch_err_sq", lambda s: float(np.sqrt(s.mean()))),
    ).reset_index()

    return {
        "total_evaluations": len(df_eval),
        "rmse_baseline": round(rmse_baseline, 4),
        "rmse_garch": round(rmse_garch, 4),
        "baseline_wins": bool(rmse_baseline < rmse_garch),
        "percentage_difference": round(abs(rmse_garch - rmse_baseline) / max(rmse_baseline, rmse_garch) * 100.0, 2),
        "ticker_summary": ticker_summary.to_dict(orient="records"),
        "detailed_results": df_eval,
    }
