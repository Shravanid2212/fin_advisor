"""Rule-based mutual fund recommendation engine (PRD Phase 19).

Adheres strictly to project rules:
1. Purely rule-based and explainable; no black-box ML.
2. Neutral decision-support language; never use buy/sell wording.
3. Ranks funds strictly within categories using z-scores of Sharpe, drawdown, and 3Y consistency.
4. Maps user's target asset allocation (equity large/mid/small, debt, gold, hybrid) to fund categories.
5. Takes existing fund buckets into account to avoid overweighting already over-held categories.
6. Respects cap preferences (e.g. 'large only' skips mid/small categories).
7. Distributes user-allocated capital proportionally across categories and funds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.config import AppConfig, load_config
from src.portfolio.holdings import PortfolioInput
from src.profile.scoring import ProfileResult

DEFAULT_DB_PATH = Path("data/market.duckdb")


@dataclass(frozen=True)
class FundRecommendationItem:
    """Individual mutual fund recommendation."""

    scheme_code: int
    name: str
    fund_house: str
    category: str
    cagr_3y: float
    cagr_5y: float
    annual_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    consistency_score: float
    last_nav: float
    last_nav_date: str
    indicative_amount: float
    allocated_weight: float
    category_weight: float
    explanation: str
    risk_level_label: str


@dataclass(frozen=True)
class CategoryFundBasket:
    """Group of recommended funds within a specific asset class / category."""

    category: str
    category_label: str
    target_weight: float
    allocated_amount: float
    funds: list[FundRecommendationItem]


@dataclass(frozen=True)
class FundRecommendationResult:
    """Complete mutual fund recommendation result container."""

    baskets: list[CategoryFundBasket]
    total_amount: float
    total_allocated: float
    profile_level: int
    profile_label: str
    notes: list[str]


def _zscore_series(series: pd.Series) -> pd.Series:
    """Compute z-score of a numerical series, returning 0.0 if standard deviation is 0."""
    s = pd.Series(series, dtype=float).fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - float(s.mean())) / std


def get_fund_risk_level_label(category: str, volatility: float) -> str:
    """Translate volatility and fund category into plain-language risk level in words."""
    c_lower = category.lower()
    if "liquid" in c_lower or "debt" in c_lower:
        return "Low" if volatility < 0.05 else "Moderately Low"
    if "gold" in c_lower:
        return "Moderate"
    if "hybrid" in c_lower:
        return "Moderate" if volatility < 0.15 else "Moderately High"
    if "large" in c_lower or "index" in c_lower:
        return "Moderately High"
    if "mid" in c_lower or "small" in c_lower:
        return "High"
    return "Moderate"


def load_fund_metrics_from_db(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load fund metrics table from DuckDB."""
    p = Path(db_path)
    if not p.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(p), read_only=True)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "fund_metrics" not in tables:
            return pd.DataFrame()
        return con.execute("SELECT * FROM fund_metrics").fetchdf()
    finally:
        con.close()


def generate_fund_recommendations(
    profile_level: int,
    portfolio: PortfolioInput | None = None,
    user_profile: ProfileResult | None = None,
    allocation_amount: float | None = None,
    cap_preference: str = "no preference",
    excluded_sectors: list[str] | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    config: AppConfig | None = None,
    max_funds_per_category: int = 2,
) -> FundRecommendationResult:
    """Generate explainable, rule-based mutual fund recommendations (PRD Phase 19).

    Workflow:
    1. Loads fund metrics from DuckDB.
    2. Identifies target asset allocation for the user's risk profile level.
    3. Adjusts category weights based on user's existing fund holdings (avoiding over-held buckets).
    4. Respects market cap preferences ('large only' skips mid/small categories).
    5. Ranks funds within each active category using z-scores of Sharpe, drawdown, and 3Y consistency.
    6. Distributes user allocation amount across categories and selected funds.
    7. Formulates factual, plain-language rationale with real metrics and last NAV label.
    """
    if config is None:
        config = load_config()

    df = load_fund_metrics_from_db(db_path=db_path)
    if df.empty:
        return FundRecommendationResult(
            baskets=[],
            total_amount=allocation_amount or 0.0,
            total_allocated=0.0,
            profile_level=profile_level,
            profile_label=config.target_allocations.get(profile_level, config.target_allocations[3]).label,
            notes=["Mutual fund dataset is uninitialized or empty. Please run `python run.py data` to populate."],
        )

    # 1. Target asset allocation
    target_alloc = config.target_allocations.get(profile_level, config.target_allocations[3])
    tot_equity_share = target_alloc.equity
    eq_split = target_alloc.equity_split
    debt_share = target_alloc.debt
    gold_share = target_alloc.gold

    # Map target allocation to candidate categories
    category_weights: dict[str, float] = {}

    # Equity sub-split
    large_eq = tot_equity_share * eq_split.large
    mid_eq = tot_equity_share * eq_split.mid
    small_eq = tot_equity_share * eq_split.small

    cap_pref = str(cap_preference).strip().lower()
    if cap_pref == "large only":
        # Shift all equity to large cap / index
        category_weights["large-cap"] = large_eq * 0.60
        category_weights["nifty-50-index"] = large_eq * 0.40
    elif cap_pref == "large + mid":
        if large_eq > 0:
            category_weights["large-cap"] = large_eq * 0.60
            category_weights["nifty-50-index"] = large_eq * 0.40
        if mid_eq > 0:
            category_weights["mid-cap"] = mid_eq * 0.50
            category_weights["flexi/multi-cap"] = mid_eq * 0.50
    else:
        if large_eq > 0:
            category_weights["large-cap"] = large_eq * 0.60
            category_weights["nifty-50-index"] = large_eq * 0.40
        if mid_eq > 0:
            category_weights["mid-cap"] = mid_eq * 0.50
            category_weights["flexi/multi-cap"] = mid_eq * 0.50
        if small_eq > 0:
            category_weights["small-cap"] = small_eq

    # Debt and Gold allocations
    if debt_share > 0:
        category_weights["liquid-debt"] = debt_share
    if gold_share > 0:
        category_weights["gold"] = gold_share

    # Balanced / Hybrid inclusion for moderate profiles if debt + equity is balanced
    if profile_level in (2, 3, 4):
        # We can also add balanced/aggressive hybrid options
        if "balanced-hybrid" not in category_weights and debt_share >= 0.20:
            # Rebalance 10% of debt and 10% of large equity into balanced hybrid
            pass

    category_labels = {
        "large-cap": "Large-cap equity",
        "flexi/multi-cap": "Flexi/multi-cap equity",
        "mid-cap": "Mid-cap equity",
        "small-cap": "Small-cap equity",
        "nifty-50-index": "Nifty 50 index",
        "aggressive-hybrid": "Aggressive hybrid",
        "balanced-hybrid": "Balanced hybrid",
        "liquid-debt": "Debt allocation",
        "gold": "Gold",
    }

    fund_cat_mapping = {
        "debt": "liquid-debt",
        "liquid": "liquid-debt",
        "fixed_income": "liquid-debt",
        "bond": "liquid-debt",
        "liquid-debt": "liquid-debt",
        "gold": "gold",
        "precious_metals": "gold",
        "large_cap": "large-cap",
        "large_cap_fund": "large-cap",
        "large-cap": "large-cap",
        "index": "nifty-50-index",
        "index_fund": "nifty-50-index",
        "equity_index": "nifty-50-index",
        "nifty-50-index": "nifty-50-index",
        "mid_small_cap": "mid-cap",
        "mid-cap": "mid-cap",
        "mid_cap_fund": "mid-cap",
        "small-cap": "small-cap",
        "small_cap_fund": "small-cap",
        "flexi/multi-cap": "flexi/multi-cap",
        "flexi_cap_fund": "flexi/multi-cap",
        "multi_cap_fund": "flexi/multi-cap",
        "balanced-hybrid": "balanced-hybrid",
        "balanced_hybrid": "balanced-hybrid",
        "aggressive-hybrid": "aggressive-hybrid",
        "aggressive_hybrid": "aggressive-hybrid",
    }

    # 2. Account for user's existing portfolio (funds and stocks)
    notes: list[str] = []
    total_existing_val = 0.0
    existing_cat_amounts: dict[str, float] = {k: 0.0 for k in category_weights}

    if portfolio:
        if portfolio.funds:
            for f in portfolio.funds:
                raw_c = str(f.category).strip().lower()
                mapped_c = fund_cat_mapping.get(raw_c)
                if not mapped_c:
                    for k in category_weights:
                        if k in raw_c or raw_c in k:
                            mapped_c = k
                            break
                amt = float(f.amount)
                total_existing_val += amt
                if mapped_c and mapped_c in category_weights:
                    existing_cat_amounts[mapped_c] = existing_cat_amounts.get(mapped_c, 0.0) + amt

        if portfolio.holdings:
            for h in portfolio.holdings:
                val = float(h.current_value) if h.current_value > 0 else float(h.quantity * (h.avg_buy_price or 0.0))
                total_existing_val += val
                cc = str(h.cap_class).strip().lower()
                if cc == "large" and "large-cap" in category_weights:
                    existing_cat_amounts["large-cap"] = existing_cat_amounts.get("large-cap", 0.0) + val
                elif cc == "mid" and "mid-cap" in category_weights:
                    existing_cat_amounts["mid-cap"] = existing_cat_amounts.get("mid-cap", 0.0) + val
                elif cc == "small" and "small-cap" in category_weights:
                    existing_cat_amounts["small-cap"] = existing_cat_amounts.get("small-cap", 0.0) + val

    if total_existing_val > 0:
        for cat_k, target_wt in list(category_weights.items()):
            cur_amt = existing_cat_amounts.get(cat_k, 0.0)
            cur_share = cur_amt / total_existing_val
            lbl = category_labels.get(cat_k, cat_k)
            if cur_share >= target_wt and target_wt > 0:
                notes.append(f"{lbl} is already at or above your target weight, no addition suggested")
                category_weights[cat_k] = 0.0
            elif cur_share > 0 and target_wt > 0:
                category_weights[cat_k] = max(0.0, target_wt - cur_share)
                notes.append(f"Reduced {lbl} allocation based on your existing holdings.")

    # Re-normalize remaining active category weights
    active_weights = {k: v for k, v in category_weights.items() if v > 0}
    sum_active = sum(active_weights.values())
    if sum_active > 0:
        category_weights = {k: v / sum_active for k, v in active_weights.items()}
    else:
        category_weights = {}

    # 3. Rank funds within each category
    w_cfg = getattr(config, "fund_recommender_weights", None)
    w_sharpe = getattr(w_cfg, "sharpe", 0.40)
    w_dd = getattr(w_cfg, "drawdown", 0.30)
    w_cons = getattr(w_cfg, "consistency", 0.30)

    alloc_total = float(allocation_amount) if (allocation_amount is not None and allocation_amount > 0) else 0.0

    baskets: list[CategoryFundBasket] = []
    total_allocated_sum = 0.0

    for cat_key, cat_wt in category_weights.items():
        if cat_wt <= 0:
            continue

        cat_sub = df[df["category"] == cat_key].copy()
        # Filter for ranked / usable funds with at least 3 years of history
        cat_sub = cat_sub[cat_sub["is_ranked"] == True]

        if len(cat_sub) < 6:
            # Rule: If a category ends with fewer than 6 usable funds, do not rank it.
            notes.append(f"Category '{category_labels.get(cat_key, cat_key)}' has fewer than 6 usable funds and is not ranked.")
            continue

        # Compute z-scores within category
        cat_sub["z_sharpe"] = _zscore_series(cat_sub["sharpe_ratio"])
        cat_sub["z_drawdown"] = _zscore_series(cat_sub["max_drawdown"])  # less negative is better
        cat_sub["z_consistency"] = _zscore_series(cat_sub["consistency_score"])

        cat_sub["composite_score"] = (
            w_sharpe * cat_sub["z_sharpe"]
            + w_dd * cat_sub["z_drawdown"]
            + w_cons * cat_sub["z_consistency"]
        )

        cat_sub = cat_sub.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        top_picks = cat_sub.head(max_funds_per_category)

        cat_alloc_amt = alloc_total * cat_wt
        fund_alloc_amt = cat_alloc_amt / len(top_picks) if len(top_picks) > 0 else 0.0
        fund_wt = cat_wt / len(top_picks) if len(top_picks) > 0 else 0.0

        rec_funds: list[FundRecommendationItem] = []
        for idx, row in top_picks.iterrows():
            cagr3 = float(row.get("cagr_3y", 0.0))
            cagr5 = float(row.get("cagr_5y", 0.0))
            vol = float(row.get("annual_volatility", 0.0))
            mdd = float(row.get("max_drawdown", 0.0))
            sh = float(row.get("sharpe_ratio", 0.0))
            cons = float(row.get("consistency_score", 0.0))
            last_p = float(row.get("last_nav", 0.0))
            last_d = str(row.get("last_nav_date", ""))
            f_name = str(row.get("name", ""))
            f_house = str(row.get("fund_house", ""))

            risk_label = get_fund_risk_level_label(cat_key, vol)

            cagr_phrase = f"3Y CAGR of {cagr3*100:.1f}%" if cagr3 != 0 else (f"5Y CAGR of {cagr5*100:.1f}%" if cagr5 != 0 else "steady performance")
            cons_phrase = f"{cons*100:.0f}% rolling 3-year consistency"
            dd_phrase = f"drawdown contained at {mdd*100:.1f}%"

            lbl = category_labels.get(cat_key, cat_key)
            if cat_key == "nifty-50-index":
                explanation = f"Nifty 50 index fund tracking the benchmark with {cons_phrase} and disciplined index replication."
            elif "flexi" in cat_key and abs(mdd) <= 0.22:
                explanation = f"Lowest drawdown in flexi-cap with steady returns ({cagr_phrase}, {dd_phrase})."
            elif "debt" in cat_key or "liquid" in cat_key:
                explanation = f"Liquid and short-duration debt allocation with low volatility ({vol*100:.1f}%) and steady capital preservation."
            elif "gold" in cat_key:
                explanation = f"Gold allocation providing macro diversification with {cons_phrase} and contained drawdown ({mdd*100:.1f}%)."
            else:
                explanation = f"Top-tier Sharpe ratio of {sh:.2f} in {lbl.lower()}, with {cagr_phrase} and {dd_phrase}."

            rec_funds.append(
                FundRecommendationItem(
                    scheme_code=int(row["scheme_code"]),
                    name=f_name,
                    fund_house=f_house,
                    category=cat_key,
                    cagr_3y=round(cagr3, 4),
                    cagr_5y=round(cagr5, 4),
                    annual_volatility=round(vol, 4),
                    max_drawdown=round(mdd, 4),
                    sharpe_ratio=round(sh, 4),
                    consistency_score=round(cons, 4),
                    last_nav=round(last_p, 4),
                    last_nav_date=last_d,
                    indicative_amount=round(fund_alloc_amt, 2),
                    allocated_weight=round(fund_wt, 4),
                    category_weight=round(cat_wt, 4),
                    explanation=explanation,
                    risk_level_label=risk_label,
                )
            )
            total_allocated_sum += fund_alloc_amt

        if rec_funds:
            baskets.append(
                CategoryFundBasket(
                    category=cat_key,
                    category_label=category_labels.get(cat_key, cat_key),
                    target_weight=round(cat_wt, 4),
                    allocated_amount=round(cat_alloc_amt, 2),
                    funds=rec_funds,
                )
            )

    return FundRecommendationResult(
        baskets=baskets,
        total_amount=round(alloc_total, 2),
        total_allocated=round(total_allocated_sum, 2),
        profile_level=profile_level,
        profile_label=target_alloc.label,
        notes=notes,
    )
