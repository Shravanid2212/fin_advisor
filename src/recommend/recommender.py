"""Recommendation engine for equity investment decision support.

Source of truth: docs/PRD.md Section 13.
Adheres strictly to AGENTS.md Hard Rules:
- Educational tool, not investment advice. Never use buy/sell wording in UI text.
- Recommender is rule-based and explainable. No black-box ML for ranking.
- All thresholds, weights, dates, and conventions come from config.yaml. No magic numbers in code.
- Deterministic: identical inputs always yield identical outputs.
- Gracefully handles empty portfolios and fewer than 8 eligible stocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from src.config import AppConfig, TargetAllocationLevel, load_config
from src.features.metrics import calculate_annualized_volatility
from src.portfolio.analytics import (
    analyze_portfolio,
    compute_portfolio_valuation,
    load_stock_daily_returns_from_db,
)
from src.portfolio.holdings import PortfolioInput, StockHolding


@dataclass
class RecommendationItem:
    """Individual recommended stock addition with explainable rationale."""

    ticker: str
    name: str
    sector: str
    cap_class: str
    risk_bucket: int
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    composite_score: float
    suggested_weight: float  # Percentage share of the suggested stock basket (sums to 1.00, <= 0.20)
    explanation: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class WhatIfComparison:
    """Before vs after portfolio impact analysis."""

    before_volatility: float
    after_volatility: float
    volatility_change: float
    before_diversification_score: float
    after_diversification_score: float
    diversification_score_change: float
    before_sector_weights: dict[str, float]
    after_sector_weights: dict[str, float]
    sector_changes: dict[str, float]


@dataclass
class LevelSensitivity:
    """Analysis of how recommendations change if risk level is shifted by +/- 1."""

    current_level: int
    level_plus_1: int | None
    level_plus_1_new_picks: list[str]
    level_plus_1_dropped_picks: list[str]
    level_plus_1_summary: str
    level_minus_1: int | None
    level_minus_1_new_picks: list[str]
    level_minus_1_dropped_picks: list[str]
    level_minus_1_summary: str


@dataclass
class AllocationItem:
    """Individual stock allocation breakdown with whole share quantities and amounts."""

    ticker: str
    name: str
    sector: str
    cap_class: str
    risk_bucket: int
    last_close: float
    price_date: str
    basket_weight: float             # Proportional share of the suggested stock basket
    indicative_amount: float         # Target allocation = total_amount * basket_weight
    suggested_quantity: int          # Whole integer shares (>= 0)
    actual_amount: float             # suggested_quantity * last_close
    min_amount_for_one_share: float  # Price needed for 1 whole share
    weight_after_addition: float     # Weight in user's total portfolio after adding this stock
    note: str = ""                   # e.g., "Minimum amount needed for 1 share is ₹..." or ""


@dataclass
class AllocationPlan:
    """Complete portfolio allocation plan for recommended stocks."""

    total_amount: float              # User-entered allocation amount
    items: list[AllocationItem]
    total_actual_amount: float       # Sum of actual_amount across all items
    unallocated_leftover: float      # total_amount - total_actual_amount (>= 0)
    new_portfolio_value: float       # Current portfolio total value + total_actual_amount
    current_portfolio_value: float   # Current portfolio total value
    target_equity_share: float       # Profile target equity fraction
    current_equity_share: float      # Current equity fraction before additions
    new_equity_share: float          # Equity fraction after additions


@dataclass
class RecommendationResult:
    """Complete output of the recommendation engine."""

    recommendations: list[RecommendationItem]
    profile_level: int
    profile_label: str
    target_equity_share: float  # Profile target equity allocation (e.g. 0.50 for Level 3)
    eligible_count: int
    total_suggested_weight: float
    what_if: WhatIfComparison
    level_sensitivity: LevelSensitivity
    allocation_plan: AllocationPlan | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cap_class_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {"large": 0, "mid": 0, "small": 0}
        for r in self.recommendations:
            c = str(r.cap_class).lower()
            mix[c] = mix.get(c, 0) + 1
        return mix


def load_stock_features_from_db(
    db_path: Path | str = "data/market.duckdb",
) -> pd.DataFrame:
    """Load stock features table from DuckDB.

    Args:
        db_path: Path to database.

    Returns:
        DataFrame containing precomputed stock features.
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database not found at: {p}")

    con = duckdb.connect(str(p), read_only=True)
    try:
        df = con.execute("SELECT * FROM stock_features").fetchdf()
        return df
    finally:
        con.close()


def _get_allowed_caps(cap_preference: str | None, profile_level: int) -> set[str]:
    """Map cap preference and profile level to allowed cap classes.

    Rules:
        Level 1: Strictly large caps only (no small or mid caps).
        Level 2: Strictly large and mid caps (no small caps).
        Level 3+: Follows user preference if provided ('large only', 'large + mid', 'include small');
                  defaults to {'large', 'mid', 'small'}.
    """
    if profile_level == 1:
        return {"large"}
    elif profile_level == 2:
        return {"large", "mid"}

    if cap_preference:
        pref = cap_preference.strip().lower()
        if "only" in pref or pref == "large":
            return {"large"}
        elif "mid" in pref and "small" not in pref:
            return {"large", "mid"}
        elif "small" in pref:
            return {"large", "mid", "small"}

    return {"large", "mid", "small"}


def _get_max_risk_bucket(profile_level: int) -> int:
    """Map profile level 1-5 to maximum permitted risk bucket (1 to 5).

    Rules:
        Level 1: max bucket 2 (no bucket above 2)
        Level 2: max bucket 2 (no bucket above 2)
        Level 3: max bucket 3 (no bucket above 3)
        Level 4: max bucket 4 (no bucket above 4)
        Level 5: max bucket 5
    """
    if profile_level <= 2:
        return 2
    elif profile_level == 3:
        return 3
    elif profile_level == 4:
        return 4
    else:
        return 5


def filter_eligible_stocks(
    features_df: pd.DataFrame,
    profile_level: int,
    cap_preference: str | None = None,
    excluded_sectors: list[str] | set[str] | None = None,
    current_portfolio_weights: dict[str, float] | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Filter stocks based on eligibility criteria from PRD Section 13.

    Filters:
    1. Cap class within profile and preference limits (Level 1 large only, Level 2 large+mid only).
    2. Volatility risk bucket not above profile maximum (Level <= 2 max 2, Level 3 max 3).
    3. Not in excluded sectors.
    4. Listing history >= min_history_years (3 years).
    5. 60-day average daily traded value >= min_liquidity_traded_value (INR 1 Crore).
    6. Current portfolio holding weight < stock concentration limit (25%).

    Args:
        features_df: DataFrame of stock features.
        profile_level: Risk profile level (1 to 5).
        cap_preference: Optional cap preference string.
        excluded_sectors: Optional set of sectors to exclude.
        current_portfolio_weights: Mapping of ticker to current equity weight.
        config: Optional AppConfig.

    Returns:
        Filtered DataFrame of eligible stocks.
    """
    if config is None:
        config = load_config()

    if features_df.empty:
        return features_df

    df = features_df.copy()
    allowed_caps = _get_allowed_caps(cap_preference, profile_level)
    max_bucket = _get_max_risk_bucket(profile_level)
    min_days = config.min_history_years * config.trading_days
    min_liquidity = config.min_liquidity_traded_value
    stock_limit = config.concentration_limits.stock

    excluded_set = {s.strip().lower() for s in (excluded_sectors or [])}
    curr_weights = current_portfolio_weights or {}

    mask = (
        df["cap_class"].isin(allowed_caps)
        & (df["risk_bucket"] <= max_bucket)
        & (~df["sector"].str.strip().str.lower().isin(excluded_set))
        & (df["trading_days"] >= min_days)
        & (df["avg_traded_value_60d"] >= min_liquidity)
        & (df["ticker"].apply(lambda t: curr_weights.get(t, 0.0) < stock_limit))
    )

    return df[mask].copy().reset_index(drop=True)


def compute_diversification_metrics(
    candidate_tickers: list[str],
    portfolio: PortfolioInput,
    returns_df: pd.DataFrame,
    test_weight: float = 0.05,
    trading_days: int = 252,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute diversification gain (volatility drop) and correlation with portfolio (or ^NSEI if empty).

    Formula:
        For portfolios with stocks:
            Aligned date index matching:
            R_port = sum(w_i * R_i(t))
            corr = Corr(R_candidate, R_port) on aligned dates
            R_sim = (1 - w_add) * R_port + w_add * R_candidate
            Delta Volatility = Vol(R_port) - Vol(R_sim)
        For portfolios with NO stocks:
            corr = Corr(R_candidate, R_NSEI) on aligned dates
            Delta Volatility = 0.01 * (1 - corr)

    Args:
        candidate_tickers: List of eligible tickers.
        portfolio: Valued user portfolio.
        returns_df: DataFrame of daily simple returns.
        test_weight: Weight of candidate addition (default 0.05 = 5%).
        trading_days: 252 trading days per year.

    Returns:
        Tuple of (delta_vol_dict, corr_with_port_dict).
    """
    delta_vols: dict[str, float] = {}
    corrs_with_port: dict[str, float] = {}

    analysed_holdings = [h for h in portfolio.holdings if h.is_analysed and h.current_value > 0]
    has_holdings = len(analysed_holdings) > 0 and not returns_df.empty

    if not has_holdings:
        # Portfolio with no stocks: use correlation with ^NSEI per PRD requirement
        nifty_col = (
            returns_df["^NSEI"].dropna()
            if "^NSEI" in returns_df.columns
            else pd.Series(dtype=float)
        )
        for t in candidate_tickers:
            if t in returns_df.columns and not nifty_col.empty:
                aligned = pd.concat([returns_df[t], nifty_col], axis=1).dropna()
                if len(aligned) > 10:
                    c = float(np.corrcoef(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)[0, 1])
                    if math.isnan(c):
                        c = 0.5
                else:
                    c = 0.5
                corrs_with_port[t] = round(c, 4)
                delta_vols[t] = round(0.01 * (1.0 - c), 6)
            else:
                corrs_with_port[t] = 0.5
                delta_vols[t] = 0.0
        return delta_vols, corrs_with_port

    # Portfolio has holdings: compute date-aligned portfolio daily return series
    avail_holdings = [h for h in analysed_holdings if h.ticker in returns_df.columns]

    if not avail_holdings:
        for t in candidate_tickers:
            delta_vols[t] = 0.0
            corrs_with_port[t] = 0.5
        return delta_vols, corrs_with_port

    tot_avail = sum(h.current_value for h in avail_holdings)
    w_vec = np.array([h.current_value / tot_avail for h in avail_holdings])
    port_tickers = [h.ticker for h in avail_holdings]

    # Date-aligned portfolio return series
    port_sub_df = returns_df[port_tickers].dropna()
    port_returns_arr = port_sub_df.values @ w_vec
    port_series = pd.Series(port_returns_arr, index=port_sub_df.index)

    for t in candidate_tickers:
        if t in returns_df.columns:
            # Strictly align dates between candidate stock and existing portfolio
            cand_series = returns_df[t].dropna()
            aligned = pd.concat([port_series, cand_series], axis=1, join="inner").dropna()

            if len(aligned) > 10:
                p_vals = aligned.iloc[:, 0].values
                s_vals = aligned.iloc[:, 1].values

                c = float(np.corrcoef(p_vals, s_vals)[0, 1])
                if math.isnan(c):
                    c = 0.5
                corrs_with_port[t] = round(c, 4)

                # Simulated portfolio with test_weight (5%) candidate addition
                curr_vol = calculate_annualized_volatility(p_vals, trading_days=trading_days)
                sim_ret = (1.0 - test_weight) * p_vals + test_weight * s_vals
                sim_vol = calculate_annualized_volatility(sim_ret, trading_days=trading_days)
                delta_vols[t] = round(curr_vol - sim_vol, 6)
            else:
                corrs_with_port[t] = 0.5
                delta_vols[t] = 0.0
        else:
            corrs_with_port[t] = 0.5
            delta_vols[t] = 0.0

    return delta_vols, corrs_with_port


def _zscore_series(s: pd.Series) -> pd.Series:
    """Compute z-score across a Series, defaulting to 0.0 if standard deviation is 0."""
    if len(s) <= 1:
        return pd.Series(0.0, index=s.index)
    std = float(s.std(ddof=1))
    if std == 0.0 or math.isnan(std):
        return pd.Series(0.0, index=s.index)
    mean = float(s.mean())
    return (s - mean) / std


def score_and_rank_stocks(
    eligible_df: pd.DataFrame,
    portfolio: PortfolioInput,
    returns_df: pd.DataFrame,
    dividend_preference: str = "no",
    favoured_sectors: list[str] | set[str] | None = None,
    config: AppConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score eligible stocks using the 5 multi-factor weights from config.yaml.

    Weights (PRD Section 13):
        35% Sharpe ratio
        20% shallow max drawdown
        25% diversification gain (drop in volatility & low correlation)
        10% fills under-represented sector vs target (overweight sectors receive zero or negative score)
        10% dividend yield (if user prefers dividends) or 12-month momentum

    Args:
        eligible_df: Filtered DataFrame of eligible stocks.
        portfolio: Valued user portfolio.
        returns_df: DataFrame of daily simple returns.
        dividend_preference: 'yes', 'no', or 'no preference'.
        config: Optional AppConfig.

    Returns:
        Tuple of (scored_df, corrs_with_port).
    """
    if config is None:
        config = load_config()

    if eligible_df.empty:
        return eligible_df, {}

    df = eligible_df.copy()
    candidate_tickers = df["ticker"].tolist()

    # 1. Diversification metrics
    delta_vols, corrs_with_port = compute_diversification_metrics(
        candidate_tickers, portfolio, returns_df, trading_days=config.trading_days
    )
    df["delta_vol"] = df["ticker"].map(delta_vols).fillna(0.0)
    df["corr_with_port"] = df["ticker"].map(corrs_with_port).fillna(0.5)
    # Diversification gain: higher delta vol and lower correlation
    df["div_gain"] = df["delta_vol"] + 0.01 * (1.0 - df["corr_with_port"])

    # 2. Sector gap vs target
    # Compute current sector weights in user portfolio
    total_equity = sum(h.current_value for h in portfolio.holdings)
    sector_weights: dict[str, float] = {}
    if total_equity > 0:
        for h in portfolio.holdings:
            if h.is_analysed and h.sector:
                sector_weights[h.sector] = sector_weights.get(h.sector, 0.0) + (h.current_value / total_equity)

    target_sector_weight = 0.10  # 10 sectors, equal target 10%
    df["current_sector_weight"] = df["sector"].map(lambda s: sector_weights.get(s, 0.0))
    # Overweight sectors receive negative gap score to actively penalize concentration
    df["sector_gap"] = df["current_sector_weight"].apply(lambda w: target_sector_weight - w)

    # 3. Dividend or Momentum factor
    use_dividend = str(dividend_preference).strip().lower() == "yes"
    if use_dividend:
        df["div_or_mom"] = df["dividend_yield"].fillna(0.0)
    else:
        df["div_or_mom"] = df["momentum_12m"].fillna(0.0)

    # 4. Standardise components using z-scores
    df["z_sharpe"] = _zscore_series(df["sharpe_ratio"])
    # Shallow max drawdown: less negative is better (higher value is better)
    df["z_drawdown"] = _zscore_series(df["max_drawdown"])
    df["z_div_gain"] = _zscore_series(df["div_gain"])
    df["z_sector_gap"] = _zscore_series(df["sector_gap"])
    df["z_div_or_mom"] = _zscore_series(df["div_or_mom"])

    # 5. Composite score
    w = config.recommender_weights
    df["composite_score"] = (
        w.sharpe * df["z_sharpe"]
        + w.drawdown * df["z_drawdown"]
        + w.diversification * df["z_div_gain"]
        + w.sector_gap * df["z_sector_gap"]
        + w.dividend_or_momentum * df["z_div_or_mom"]
    )

    # 6. Favoured sector bonus (applied to composite score)
    fav_bonus = getattr(config.recommender, "favoured_sector_bonus", 0.5) if config else 0.5
    fav_set = {str(s).strip().lower() for s in (favoured_sectors or [])}
    df["is_favoured"] = df["sector"].apply(lambda s: str(s).strip().lower() in fav_set)
    df["fav_bonus_val"] = df["is_favoured"].apply(lambda fav: fav_bonus if fav else 0.0)
    df["composite_score"] = df["composite_score"] + df["fav_bonus_val"]

    df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    return df, corrs_with_port


def select_top_candidates(
    scored_df: pd.DataFrame,
    max_picks: int = 8,
    max_per_sector: int = 2,
    per_stock_weight_cap: float = 0.20,
) -> list[dict[str, Any]]:
    """Select top recommendations enforcing at most max_per_sector and computing inverse-vol weights.

    Formula for inverse-volatility weights:
        w_raw_i = (1 / sigma_i) / sum(1 / sigma_j)
        Summing to 100% (1.00) of the suggested stock basket.
        Capped at per_stock_weight_cap (20% = 0.20) from config.yaml.
        Weights differ between stocks based on their volatility.

    Args:
        scored_df: Scored and sorted candidate DataFrame.
        max_picks: Maximum number of recommendations (default 8).
        max_per_sector: Maximum recommendations per sector (default 2).
        per_stock_weight_cap: Maximum weight allocated to any single recommendation (default 0.20).

    Returns:
        List of selected candidate dicts with capped inverse-volatility weights summing to 1.00.
    """
    if scored_df.empty:
        return []

    selected: list[dict[str, Any]] = []
    sector_counts: dict[str, int] = {}

    for _, row in scored_df.iterrows():
        sec = row["sector"]
        if sector_counts.get(sec, 0) < max_per_sector:
            selected.append(row.to_dict())
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= max_picks:
                break

    if not selected:
        return []

    # Inverse-volatility weighting summing to 100% (1.00) across the picks
    inv_vols = [1.0 / max(0.01, float(s["annual_volatility"])) for s in selected]
    tot_inv = sum(inv_vols)
    weights = np.array([iv / tot_inv for iv in inv_vols])

    # Apply per-stock cap (default 20% = 0.20) with iterative proportional redistribution
    cap = per_stock_weight_cap
    if len(selected) * cap >= 1.0:
        for _ in range(20):
            excess = np.sum(np.maximum(0.0, weights - cap))
            if excess <= 1e-6:
                break
            weights = np.minimum(weights, cap)
            uncapped_mask = weights < (cap - 1e-6)
            if not np.any(uncapped_mask):
                break
            uncapped_sum = np.sum(weights[uncapped_mask])
            if uncapped_sum > 0:
                weights[uncapped_mask] += excess * (weights[uncapped_mask] / uncapped_sum)
        # Strictly normalize to 1.00
        weights = weights / np.sum(weights)
    else:
        weights = np.minimum(weights, cap)

    for i, s in enumerate(selected):
        s["suggested_weight"] = round(float(weights[i]), 4)

    return selected


def build_explanation(
    item: dict[str, Any],
    eligible_df: pd.DataFrame,
    portfolio: PortfolioInput,
    favoured_sectors: list[str] | set[str] | None = None,
    cluster_map: dict[str, str] | None = None,
) -> str:
    """Generate an explainable, fact-based rationale string for a recommendation.

    Rule: Never says 'adds exposure' for an overweight sector.
    """
    sharpe = float(item.get("sharpe_ratio", 0.0))
    sec = item.get("sector", "")
    corr = float(item.get("corr_with_port", 0.0))
    ticker = item.get("ticker", "")

    # Sharpe percentile among eligible stocks
    all_sharpes = eligible_df["sharpe_ratio"].dropna().values
    if len(all_sharpes) > 1:
        rank_pct = round(100.0 - stats.percentileofscore(all_sharpes, sharpe), 0)
        sharpe_desc = f"top {max(1, int(rank_pct))}% of eligible"
    else:
        sharpe_desc = "eligible pick"

    # Current sector exposure
    tot_equity = sum(h.current_value for h in portfolio.holdings)
    curr_sec_val = sum(h.current_value for h in portfolio.holdings if h.is_analysed and h.sector == sec)
    curr_sec_pct = round((curr_sec_val / tot_equity) * 100.0, 1) if tot_equity > 0 else 0.0

    if curr_sec_pct == 0.0:
        sector_rationale = f"adds exposure to {sec} where you hold 0% (target 10%)"
    elif curr_sec_pct < 10.0:
        sector_rationale = f"increases underweight exposure in {sec} from {curr_sec_pct:.0f}% toward target 10%"
    else:
        # Overweight sector: never says "adds exposure"
        sector_rationale = f"maintains exposure in {sec} (currently {curr_sec_pct:.0f}% vs target 10%)"

    fav_set = {str(s).strip().lower() for s in (favoured_sectors or [])}
    pref_clause = f" Matches your preference for {sec}." if str(sec).strip().lower() in fav_set else ""

    cluster_clause = ""
    if cluster_map and ticker in cluster_map:
        c_name = cluster_map[ticker]
        overlapping_holdings = [
            h.ticker for h in portfolio.holdings
            if h.is_analysed and cluster_map.get(h.ticker) == c_name and h.ticker != ticker
        ]
        if overlapping_holdings:
            cluster_clause = f" Behaviour group: {c_name} (shared with {', '.join(overlapping_holdings[:2])} in your portfolio)."
        else:
            cluster_clause = f" Behaviour group: {c_name}."

    return (
        f"Why this stock: Sharpe {sharpe:.2f} ({sharpe_desc}), correlation with your portfolio {corr:.2f}, "
        f"{sector_rationale}.{pref_clause}{cluster_clause}"
    )


def compute_what_if_analysis(
    portfolio: PortfolioInput,
    recommendations: list[RecommendationItem],
    returns_df: pd.DataFrame,
    db_path: Path | str = "data/market.duckdb",
    trading_days: int = 252,
    allocation_items: list[AllocationItem] | None = None,
) -> WhatIfComparison:
    """Compare portfolio volatility, diversification, and sector exposures before and after adding recommendations.

    Args:
        portfolio: Valued user portfolio.
        recommendations: Suggested recommendation items with suggested_weight.
        returns_df: Daily returns DataFrame.
        db_path: Database path.
        trading_days: 252.
        allocation_items: Optional list of AllocationItem containing rounded integer share quantities.

    Returns:
        WhatIfComparison object detailing before, after, and delta metrics.
    """
    base_analytics = analyze_portfolio(
        portfolio, returns_df=returns_df, db_path=db_path, trading_days=trading_days
    )

    before_vol = base_analytics.volatility
    before_div = base_analytics.diversification_score
    before_sectors = base_analytics.sector_weights.copy()

    if not recommendations:
        return WhatIfComparison(
            before_volatility=before_vol,
            after_volatility=before_vol,
            volatility_change=0.0,
            before_diversification_score=before_div,
            after_diversification_score=before_div,
            diversification_score_change=0.0,
            before_sector_weights=before_sectors,
            after_sector_weights=before_sectors,
            sector_changes={},
        )

    # Construct hypothetical combined portfolio
    new_holdings = [
        StockHolding(
            ticker=h.ticker,
            quantity=h.quantity,
            avg_buy_price=h.avg_buy_price,
            is_analysed=h.is_analysed,
            name=h.name,
            sector=h.sector,
            cap_class=h.cap_class,
            latest_price=h.latest_price,
            current_value=h.current_value,
        )
        for h in portfolio.holdings
    ]

    base_equity = base_analytics.equity_value
    reference_equity = base_equity if base_equity > 0 else 100000.0

    # Add suggested recommendations
    if allocation_items and any(item.suggested_quantity > 0 for item in allocation_items):
        existing_map = {h.ticker: h for h in new_holdings if h.is_analysed}
        for item in allocation_items:
            if item.suggested_quantity > 0:
                if item.ticker in existing_map:
                    h = existing_map[item.ticker]
                    h.quantity += item.suggested_quantity
                    h.current_value += item.actual_amount
                else:
                    new_holdings.append(
                        StockHolding(
                            ticker=item.ticker,
                            quantity=float(item.suggested_quantity),
                            avg_buy_price=item.last_close,
                            is_analysed=True,
                            name=item.name,
                            sector=item.sector,
                            cap_class=item.cap_class,
                            latest_price=item.last_close,
                            current_value=item.actual_amount,
                        )
                    )
    else:
        # Fallback to proportional addition if no whole share quantities provided
        for rec in recommendations:
            add_val = reference_equity * (rec.suggested_weight * 0.5)
            new_holdings.append(
                StockHolding(
                    ticker=rec.ticker,
                    quantity=add_val / max(1.0, rec.metrics.get("latest_price", 100.0)),
                    avg_buy_price=rec.metrics.get("latest_price", 100.0),
                    is_analysed=True,
                    name=rec.name,
                    sector=rec.sector,
                    cap_class=rec.cap_class,
                    latest_price=rec.metrics.get("latest_price", 100.0),
                    current_value=add_val,
                )
            )

    sim_portfolio = PortfolioInput(holdings=new_holdings, funds=portfolio.funds)
    sim_tickers = [h.ticker for h in new_holdings if h.is_analysed]
    sim_returns_df = load_stock_daily_returns_from_db(sim_tickers, db_path=db_path)
    if sim_returns_df.empty:
        sim_returns_df = returns_df

    sim_analytics = analyze_portfolio(
        sim_portfolio, returns_df=sim_returns_df, db_path=db_path, trading_days=trading_days
    )

    after_vol = sim_analytics.volatility
    after_div = sim_analytics.diversification_score
    after_sectors = sim_analytics.sector_weights.copy()

    all_secs = set(before_sectors.keys()).union(after_sectors.keys())
    sector_changes = {
        s: round(after_sectors.get(s, 0.0) - before_sectors.get(s, 0.0), 4) for s in all_secs
    }

    return WhatIfComparison(
        before_volatility=round(before_vol, 4),
        after_volatility=round(after_vol, 4),
        volatility_change=round(after_vol - before_vol, 4),
        before_diversification_score=round(before_div, 2),
        after_diversification_score=round(after_div, 2),
        diversification_score_change=round(after_div - before_div, 2),
        before_sector_weights=before_sectors,
        after_sector_weights=after_sectors,
        sector_changes=sector_changes,
    )


def compute_recommendation_allocations(
    recommendations: list[RecommendationItem],
    total_amount: float,
    portfolio: PortfolioInput | None = None,
    current_portfolio_value: float = 0.0,
    target_equity_share: float = 0.50,
    current_equity_value: float = 0.0,
    db_path: Path | str = "data/market.duckdb",
    per_stock_weight_cap: float = 0.20,
) -> AllocationPlan:
    """Compute share quantities, indicative & actual amounts, leftovers, and redistribution.

    Formulas & Rules:
    - indicative_amount = total_amount * basket_weight
    - suggested_quantity = floor(indicative_amount / last_close)
    - actual_amount = suggested_quantity * last_close
    - unallocated_leftover = total_amount - sum(actual_amount)
    - If quantity is 0 for a pick:
      - minimum amount needed for one share is shown (= last_close).
      - that pick's leftover is redistributed to other picks by their weights where it buys
        whole shares, without breaking the per-stock weight cap (default 20%).

    Args:
        recommendations: Suggested stock ideas.
        total_amount: Total investment amount in rupees.
        portfolio: User's portfolio input.
        current_portfolio_value: Total value of current portfolio.
        target_equity_share: Target equity allocation fraction.
        current_equity_value: Current equity value of portfolio.
        db_path: Path to DuckDB database.
        per_stock_weight_cap: Max allowable weight per stock in allocated basket (default 0.20).

    Returns:
        AllocationPlan containing per-pick breakdowns and portfolio totals.
    """
    # Derive portfolio valuation if not provided directly
    if portfolio and current_portfolio_value == 0.0:
        current_portfolio_value = sum(
            float(h.current_value) for h in portfolio.holdings if h.is_analysed
        ) + sum(float(f.amount) for f in portfolio.funds)
        current_equity_value = sum(
            float(h.current_value) for h in portfolio.holdings if h.is_analysed
        )

    if total_amount <= 0 or not recommendations:
        return AllocationPlan(
            total_amount=max(0.0, float(total_amount)),
            items=[],
            total_actual_amount=0.0,
            unallocated_leftover=max(0.0, float(total_amount)),
            new_portfolio_value=current_portfolio_value,
            current_portfolio_value=current_portfolio_value,
            target_equity_share=target_equity_share,
            current_equity_share=(current_equity_value / current_portfolio_value) if current_portfolio_value > 0 else 0.0,
            new_equity_share=(current_equity_value / current_portfolio_value) if current_portfolio_value > 0 else 0.0,
        )

    # 1. Fetch latest clean close prices and dates from prices table
    tickers = [r.ticker for r in recommendations]
    price_map: dict[str, tuple[float, str]] = {}
    try:
        p = Path(db_path)
        if p.exists():
            con = duckdb.connect(str(p), read_only=True)
            placeholders = ",".join(f"'{t}'" for t in tickers)
            query = f"""
                SELECT ticker, close, date
                FROM (
                    SELECT ticker, close, date,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) as rn
                    FROM prices
                    WHERE ticker IN ({placeholders}) AND is_filled = FALSE
                ) WHERE rn = 1
            """
            df_p = con.execute(query).fetchdf()
            con.close()
            for _, row in df_p.iterrows():
                price_map[str(row["ticker"])] = (float(row["close"]), str(row["date"]))
    except Exception:
        pass

    # 2. Initial pass: compute indicative amounts, floor to whole shares
    n_picks = len(recommendations)
    last_closes: list[float] = []
    price_dates: list[str] = []
    basket_weights: list[float] = []
    indicative_amounts: list[float] = []
    suggested_quantities: list[int] = []
    actual_amounts: list[float] = []

    for r in recommendations:
        p_info = price_map.get(r.ticker, (float(r.metrics.get("latest_price", 100.0)), "Latest"))
        price = p_info[0] if p_info[0] > 0 else 100.0
        p_date = p_info[1]
        w = float(r.suggested_weight)

        indicative = float(total_amount * w)
        qty = int(math.floor(indicative / price))
        act = float(qty * price)

        last_closes.append(price)
        price_dates.append(p_date)
        basket_weights.append(w)
        indicative_amounts.append(indicative)
        suggested_quantities.append(qty)
        actual_amounts.append(act)

    # 3. Redistribution of leftover from 0-quantity picks to other picks by weight
    total_spent = sum(actual_amounts)
    remaining_leftover = total_amount - total_spent

    # Candidates eligible for redistribution: picks with non-zero allocation or those that can buy shares,
    # ordered by original basket weights descending
    candidate_indices = sorted(
        range(n_picks),
        key=lambda idx: basket_weights[idx],
        reverse=True,
    )

    # Redistribute where it buys whole shares without breaking the per-stock cap
    changed = True
    passes = 0
    while changed and passes < 50 and remaining_leftover > 0:
        changed = False
        passes += 1
        for j in candidate_indices:
            pj = last_closes[j]
            if pj <= remaining_leftover:
                new_act = actual_amounts[j] + pj
                # Enforce per-stock cap (allow tiny floating epsilon 1e-4)
                if (new_act / total_amount) <= (per_stock_weight_cap + 1e-4):
                    suggested_quantities[j] += 1
                    actual_amounts[j] = new_act
                    remaining_leftover -= pj
                    changed = True

    # 4. Final totals & post-addition portfolio weights
    total_actual_amount = round(float(sum(actual_amounts)), 2)
    unallocated_leftover = round(max(0.0, float(total_amount - total_actual_amount)), 2)

    existing_stock_values: dict[str, float] = {}
    if portfolio and portfolio.holdings:
        for h in portfolio.holdings:
            if h.is_analysed:
                existing_stock_values[h.ticker] = existing_stock_values.get(h.ticker, 0.0) + float(h.current_value)

    new_port_val = round(float(current_portfolio_value + total_actual_amount), 2)
    new_equity_val = round(float(current_equity_value + total_actual_amount), 2)
    new_eq_share = (new_equity_val / new_port_val) if new_port_val > 0 else 0.0
    curr_eq_share = (current_equity_value / current_portfolio_value) if current_portfolio_value > 0 else 0.0

    items: list[AllocationItem] = []
    for i, r in enumerate(recommendations):
        qty = suggested_quantities[i]
        act = round(float(actual_amounts[i]), 2)
        p = round(float(last_closes[i]), 2)
        ind = round(float(indicative_amounts[i]), 2)

        if qty == 0:
            note = f"Minimum amount needed for 1 share is ₹{p:,.2f}"
        else:
            note = ""

        # Weight of the user's total portfolio after adding this stock
        exist_v = existing_stock_values.get(r.ticker, 0.0)
        stock_new_total = exist_v + act
        if new_port_val > 0:
            port_wt_after = round(stock_new_total / new_port_val, 4)
        elif total_actual_amount > 0:
            port_wt_after = round(act / total_actual_amount, 4)
        else:
            port_wt_after = 0.0

        items.append(
            AllocationItem(
                ticker=r.ticker,
                name=r.name,
                sector=r.sector,
                cap_class=r.cap_class,
                risk_bucket=r.risk_bucket,
                last_close=p,
                price_date=price_dates[i],
                basket_weight=round(float(r.suggested_weight), 4),
                indicative_amount=ind,
                suggested_quantity=qty,
                actual_amount=act,
                min_amount_for_one_share=p,
                weight_after_addition=port_wt_after,
                note=note,
            )
        )

    return AllocationPlan(
        total_amount=round(float(total_amount), 2),
        items=items,
        total_actual_amount=total_actual_amount,
        unallocated_leftover=unallocated_leftover,
        new_portfolio_value=new_port_val,
        current_portfolio_value=round(float(current_portfolio_value), 2),
        target_equity_share=round(float(target_equity_share), 4),
        current_equity_share=round(float(curr_eq_share), 4),
        new_equity_share=round(float(new_eq_share), 4),
    )


def compute_level_sensitivity(
    portfolio: PortfolioInput,
    current_level: int,
    cap_preference: str | None = None,
    excluded_sectors: list[str] | None = None,
    favoured_sectors: list[str] | None = None,
    dividend_preference: str = "no",
    features_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    db_path: Path | str = "data/market.duckdb",
    config: AppConfig | None = None,
) -> LevelSensitivity:
    """Analyze which stock picks change if the investor's risk profile level is shifted by +1 or -1.

    Generates explanations strictly from the actual differences between the recommendation sets.
    """
    if config is None:
        config = load_config()

    current_res = generate_recommendations(
        portfolio=portfolio,
        profile_level=current_level,
        cap_preference=cap_preference,
        excluded_sectors=excluded_sectors,
        favoured_sectors=favoured_sectors,
        dividend_preference=dividend_preference,
        features_df=features_df,
        returns_df=returns_df,
        db_path=db_path,
        config=config,
        include_sensitivity=False,
    )
    current_picks = {r.ticker for r in current_res.recommendations}

    # Level +1
    plus_1_lvl = current_level + 1 if current_level < 5 else None
    new_plus_1: list[str] = []
    dropped_plus_1: list[str] = []
    summary_plus_1 = ""

    if plus_1_lvl is not None:
        res_plus_1 = generate_recommendations(
            portfolio=portfolio,
            profile_level=plus_1_lvl,
            cap_preference=cap_preference,
            excluded_sectors=excluded_sectors,
            favoured_sectors=favoured_sectors,
            dividend_preference=dividend_preference,
            features_df=features_df,
            returns_df=returns_df,
            db_path=db_path,
            config=config,
            include_sensitivity=False,
        )
        picks_plus_1 = {r.ticker for r in res_plus_1.recommendations}
        new_plus_1 = sorted(list(picks_plus_1 - current_picks))
        dropped_plus_1 = sorted(list(current_picks - picks_plus_1))

        if new_plus_1 or dropped_plus_1:
            new_feats = (
                features_df[features_df["ticker"].isin(new_plus_1)]
                if not features_df.empty
                else pd.DataFrame()
            )
            new_vols = (
                new_feats["annual_volatility"].mean() if not new_feats.empty else 0.0
            )
            new_buckets = (
                sorted(list(set(new_feats["risk_bucket"].tolist())))
                if not new_feats.empty
                else []
            )
            dropped_feats = (
                features_df[features_df["ticker"].isin(dropped_plus_1)]
                if not features_df.empty
                else pd.DataFrame()
            )
            dropped_vols = (
                dropped_feats["annual_volatility"].mean()
                if not dropped_feats.empty
                else 0.0
            )

            parts = []
            if new_plus_1:
                parts.append(
                    f"adds {', '.join(new_plus_1)} (avg vol {new_vols*100:.1f}%, risk buckets {new_buckets})"
                )
            if dropped_plus_1:
                parts.append(
                    f"drops {', '.join(dropped_plus_1)} (avg vol {dropped_vols*100:.1f}%)"
                )
            summary_plus_1 = f"At Level {plus_1_lvl}: {' and '.join(parts)}."
        else:
            summary_plus_1 = f"At Level {plus_1_lvl}: recommendation list is identical."
    else:
        summary_plus_1 = "Already at maximum Aggressive risk level (Level 5)."

    # Level -1
    minus_1_lvl = current_level - 1 if current_level > 1 else None
    new_minus_1: list[str] = []
    dropped_minus_1: list[str] = []
    summary_minus_1 = ""

    if minus_1_lvl is not None:
        res_minus_1 = generate_recommendations(
            portfolio=portfolio,
            profile_level=minus_1_lvl,
            cap_preference=cap_preference,
            excluded_sectors=excluded_sectors,
            favoured_sectors=favoured_sectors,
            dividend_preference=dividend_preference,
            features_df=features_df,
            returns_df=returns_df,
            db_path=db_path,
            config=config,
            include_sensitivity=False,
        )
        picks_minus_1 = {r.ticker for r in res_minus_1.recommendations}
        new_minus_1 = sorted(list(picks_minus_1 - current_picks))
        dropped_minus_1 = sorted(list(current_picks - picks_minus_1))

        if new_minus_1 or dropped_minus_1:
            new_feats = (
                features_df[features_df["ticker"].isin(new_minus_1)]
                if not features_df.empty
                else pd.DataFrame()
            )
            new_vols = (
                new_feats["annual_volatility"].mean() if not new_feats.empty else 0.0
            )
            new_buckets = (
                sorted(list(set(new_feats["risk_bucket"].tolist())))
                if not new_feats.empty
                else []
            )
            dropped_feats = (
                features_df[features_df["ticker"].isin(dropped_minus_1)]
                if not features_df.empty
                else pd.DataFrame()
            )
            dropped_vols = (
                dropped_feats["annual_volatility"].mean()
                if not dropped_feats.empty
                else 0.0
            )

            parts = []
            if dropped_minus_1:
                parts.append(
                    f"drops {', '.join(dropped_minus_1)} (avg vol {dropped_vols*100:.1f}%)"
                )
            if new_minus_1:
                parts.append(
                    f"adds {', '.join(new_minus_1)} (avg vol {new_vols*100:.1f}%, risk buckets {new_buckets})"
                )
            summary_minus_1 = f"At Level {minus_1_lvl}: {' and '.join(parts)}."
        else:
            summary_minus_1 = f"At Level {minus_1_lvl}: recommendation list is identical."
    else:
        summary_minus_1 = "Already at minimum Conservative risk level (Level 1)."

    return LevelSensitivity(
        current_level=current_level,
        level_plus_1=plus_1_lvl,
        level_plus_1_new_picks=new_plus_1,
        level_plus_1_dropped_picks=dropped_plus_1,
        level_plus_1_summary=summary_plus_1,
        level_minus_1=minus_1_lvl,
        level_minus_1_new_picks=new_minus_1,
        level_minus_1_dropped_picks=dropped_minus_1,
        level_minus_1_summary=summary_minus_1,
    )


def generate_recommendations(
    portfolio: PortfolioInput,
    profile_level: int = 3,
    cap_preference: str | None = None,
    excluded_sectors: list[str] | set[str] | None = None,
    favoured_sectors: list[str] | set[str] | None = None,
    dividend_preference: str = "no",
    features_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    db_path: Path | str = "data/market.duckdb",
    config: AppConfig | None = None,
    include_sensitivity: bool = True,
    allocation_amount: float | None = None,
) -> RecommendationResult:
    """Generate explainable, rule-based equity recommendations according to PRD Section 13.

    Workflow:
    1. Filter eligible universe (cap preference, risk bucket, excluded sectors, liquidity, listing history).
    2. Multi-factor scoring (Sharpe, shallow drawdown, diversification gain, sector gap, dividend/momentum).
    3. Select top 8 candidates (at most 2 per sector).
    4. Inverse-volatility weighting with strict per-stock cap (default 20% in suggested basket).
    5. Formulate verifiable, numerical rationale explanation per stock.
    6. Run before/after what-if volatility and sector analysis.
    7. Evaluate level sensitivity (+/- 1 risk level changes).

    Args:
        portfolio: Validated user portfolio input.
        profile_level: User's risk level (1 to 5).
        cap_preference: 'large only', 'large + mid', or 'include small'.
        excluded_sectors: List/set of sectors to avoid.
        dividend_preference: 'yes', 'no', or 'no preference'.
        features_df: Pre-loaded stock features DataFrame.
        returns_df: Pre-loaded returns DataFrame.
        db_path: Path to DuckDB database.
        config: Optional AppConfig.
        include_sensitivity: Whether to compute +/-1 level sensitivity.

    Returns:
        RecommendationResult container.
    """
    if config is None:
        config = load_config()

    if features_df is None:
        features_df = load_stock_features_from_db(db_path=db_path)

    # 1. Ensure portfolio holdings are valued at latest market prices
    compute_portfolio_valuation(portfolio, db_path=db_path)
    tot_equity = sum(h.current_value for h in portfolio.holdings)
    curr_weights = {
        h.ticker: (h.current_value / tot_equity) for h in portfolio.holdings if h.is_analysed and tot_equity > 0
    }

    # 2. Filter eligible stocks
    eligible_df = filter_eligible_stocks(
        features_df=features_df,
        profile_level=profile_level,
        cap_preference=cap_preference,
        excluded_sectors=excluded_sectors,
        current_portfolio_weights=curr_weights,
        config=config,
    )

    # 2b. ML Risk Outlook filter: Profile levels 1 and 2 only exclude stocks with 'Risk may be rising'
    ml_cfg = getattr(config, "ml", None)
    ro_enabled = getattr(ml_cfg.risk_outlook, "enabled", False) if ml_cfg else False
    ml_skipped_notes: list[str] = []

    if profile_level in (1, 2) and ro_enabled:
        try:
            con_ml = duckdb.connect(str(db_path), read_only=True)
            tables = con_ml.execute("SHOW TABLES").df()["name"].tolist()
            if "risk_outlook" in tables:
                rising_tickers = set(
                    con_ml.execute(
                        "SELECT ticker FROM risk_outlook WHERE outlook = 'Risk may be rising'"
                    ).df()["ticker"].tolist()
                )
                skipped_tickers = [t for t in eligible_df["ticker"] if t in rising_tickers]
                if skipped_tickers:
                    eligible_df = eligible_df[~eligible_df["ticker"].isin(rising_tickers)].copy().reset_index(drop=True)
                    ml_skipped_notes.append(
                        f"Skipped {len(skipped_tickers)} candidate(s) ({', '.join(skipped_tickers)}) flagged with forward 'Risk may be rising' for risk level {profile_level}."
                    )
            con_ml.close()
        except Exception:
            pass

    # 2c. Load cluster map for explainability if clustering is enabled
    cl_enabled = getattr(ml_cfg.clusters, "enabled", False) if ml_cfg else False
    cluster_map: dict[str, str] = {}
    if cl_enabled:
        try:
            con_cl = duckdb.connect(str(db_path), read_only=True)
            tables = con_cl.execute("SHOW TABLES").df()["name"].tolist()
            if "stock_clusters" in tables:
                df_sc = con_cl.execute("SELECT ticker, cluster_name FROM stock_clusters").df()
                cluster_map = dict(zip(df_sc["ticker"], df_sc["cluster_name"]))
            con_cl.close()
        except Exception:
            pass

    target_alloc = config.target_allocations[profile_level]
    target_equity_share = target_alloc.equity

    if eligible_df.empty:
        empty_what_if = WhatIfComparison(
            before_volatility=0.0,
            after_volatility=0.0,
            volatility_change=0.0,
            before_diversification_score=0.0,
            after_diversification_score=0.0,
            diversification_score_change=0.0,
            before_sector_weights={},
            after_sector_weights={},
            sector_changes={},
        )
        empty_sensitivity = LevelSensitivity(
            current_level=profile_level,
            level_plus_1=None,
            level_plus_1_new_picks=[],
            level_plus_1_dropped_picks=[],
            level_plus_1_summary="No eligible stocks.",
            level_minus_1=None,
            level_minus_1_new_picks=[],
            level_minus_1_dropped_picks=[],
            level_minus_1_summary="No eligible stocks.",
        )
        return RecommendationResult(
            recommendations=[],
            profile_level=profile_level,
            profile_label=target_alloc.label,
            target_equity_share=target_equity_share,
            eligible_count=0,
            total_suggested_weight=0.0,
            what_if=empty_what_if,
            level_sensitivity=empty_sensitivity,
            notes=["No stocks matched the eligibility criteria for the given preferences."],
        )

    # 3. Load daily returns for candidate and portfolio stocks (and ^NSEI for benchmark alignment)
    if returns_df is None:
        needed_tickers = list(
            set(
                eligible_df["ticker"].tolist()
                + [h.ticker for h in portfolio.holdings if h.is_analysed]
                + ["^NSEI"]
            )
        )
        returns_df = load_stock_daily_returns_from_db(needed_tickers, db_path=db_path)

    # 4. Multi-factor scoring
    fav_set_clean = [str(s).strip() for s in (favoured_sectors or [])]
    eligible_sectors = set(eligible_df["sector"].unique()) if not eligible_df.empty else set()
    fav_notes: list[str] = []
    for f_sec in fav_set_clean:
        if f_sec not in eligible_sectors:
            fav_notes.append(f"No stocks in {f_sec} fit your risk level")

    scored_df, corrs_with_port = score_and_rank_stocks(
        eligible_df=eligible_df,
        portfolio=portfolio,
        returns_df=returns_df,
        dividend_preference=dividend_preference,
        favoured_sectors=favoured_sectors,
        config=config,
    )

    # 5. Select top candidates (up to max_recommendations, max 2 per sector)
    per_stock_cap = config.recommender.per_stock_weight_cap
    max_recs = config.recommender.max_recommendations
    max_per_sec = config.recommender.max_per_sector

    selected_dicts = select_top_candidates(
        scored_df,
        max_picks=max_recs,
        max_per_sector=max_per_sec,
        per_stock_weight_cap=per_stock_cap,
    )

    # 6. Build RecommendationItem objects with factual explanations
    rec_items: list[RecommendationItem] = []
    for s in selected_dicts:
        exp = build_explanation(
            s,
            eligible_df,
            portfolio,
            favoured_sectors=favoured_sectors,
            cluster_map=cluster_map,
        )
        rec_items.append(
            RecommendationItem(
                ticker=s["ticker"],
                name=s["name"],
                sector=s["sector"],
                cap_class=s["cap_class"],
                risk_bucket=int(s["risk_bucket"]),
                annual_volatility=round(float(s["annual_volatility"]), 4),
                sharpe_ratio=round(float(s["sharpe_ratio"]), 4),
                max_drawdown=round(float(s["max_drawdown"]), 4),
                composite_score=round(float(s["composite_score"]), 4),
                suggested_weight=float(s["suggested_weight"]),
                explanation=exp,
                metrics={
                    "sharpe_ratio": round(float(s["sharpe_ratio"]), 4),
                    "annual_volatility": round(float(s["annual_volatility"]), 4),
                    "risk_bucket": int(s["risk_bucket"]),
                    "max_drawdown": round(float(s["max_drawdown"]), 4),
                    "corr_with_port": round(float(s.get("corr_with_port", 0.0)), 4),
                    "sector_gap": round(float(s.get("sector_gap", 0.0)), 4),
                    "div_or_mom": round(float(s.get("div_or_mom", 0.0)), 4),
                },
            )
        )

    # 7. Allocation Plan (if allocation amount is provided)
    allocation_plan = None
    if allocation_amount is not None and allocation_amount > 0:
        base_analytics = analyze_portfolio(portfolio, db_path=db_path)
        allocation_plan = compute_recommendation_allocations(
            recommendations=rec_items,
            total_amount=allocation_amount,
            portfolio=portfolio,
            current_portfolio_value=base_analytics.total_value,
            target_equity_share=target_equity_share,
            current_equity_value=base_analytics.equity_value,
            db_path=db_path,
            per_stock_weight_cap=per_stock_cap,
        )

    # 8. What-if analysis (using rounded integer quantities if allocated)
    what_if = compute_what_if_analysis(
        portfolio=portfolio,
        recommendations=rec_items,
        returns_df=returns_df,
        db_path=db_path,
        trading_days=config.trading_days,
        allocation_items=allocation_plan.items if allocation_plan else None,
    )

    # 9. Level sensitivity
    if include_sensitivity:
        level_sens = compute_level_sensitivity(
            portfolio=portfolio,
            current_level=profile_level,
            cap_preference=cap_preference,
            excluded_sectors=list(excluded_sectors) if excluded_sectors else None,
            favoured_sectors=list(favoured_sectors) if favoured_sectors else None,
            dividend_preference=dividend_preference,
            features_df=features_df,
            returns_df=returns_df,
            db_path=db_path,
            config=config,
        )
    else:
        level_sens = LevelSensitivity(
            current_level=profile_level,
            level_plus_1=None,
            level_plus_1_new_picks=[],
            level_plus_1_dropped_picks=[],
            level_plus_1_summary="",
            level_minus_1=None,
            level_minus_1_new_picks=[],
            level_minus_1_dropped_picks=[],
            level_minus_1_summary="",
        )

    tot_sugg_weight = round(sum(r.suggested_weight for r in rec_items), 4)

    all_notes = [
        f"Identified {len(rec_items)} suggested additions from {len(eligible_df)} eligible candidates.",
        f"Weights sum to {tot_sugg_weight*100:.1f}% of suggested stock basket, capped at {per_stock_cap*100:.0f}% per stock.",
        f"Target equity share for {target_alloc.label} (Level {profile_level}) is {target_equity_share*100:.0f}%.",
        "Educational decision support only. Not investment advice.",
    ] + fav_notes + ml_skipped_notes

    return RecommendationResult(
        recommendations=rec_items,
        profile_level=profile_level,
        profile_label=target_alloc.label,
        target_equity_share=target_equity_share,
        eligible_count=len(eligible_df),
        total_suggested_weight=tot_sugg_weight,
        what_if=what_if,
        level_sensitivity=level_sens,
        allocation_plan=allocation_plan,
        notes=all_notes,
    )
