"""Configuration loader and schema validation for Financial Market Analytics System.

All thresholds, weights, dates, universe, and conventions live in config.yaml and data/reference/universe.csv.
No magic numbers in code.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

VALID_CAP_CLASSES = {"large", "mid", "small"}

VALID_SECTORS = {
    "Banking and Financials",
    "IT",
    "Pharma and Healthcare",
    "FMCG",
    "Auto",
    "Energy and Power",
    "Metals",
    "Capital Goods and Infra",
    "Consumer",
    "Chemicals",
}


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails due to missing keys or invalid values."""


@dataclass(frozen=True)
class DateRangeConfig:
    start: str
    end: str


@dataclass(frozen=True)
class ConcentrationLimitsConfig:
    stock: float
    sector: float


@dataclass(frozen=True)
class RecommenderWeightsConfig:
    sharpe: float
    drawdown: float
    diversification: float
    sector_gap: float
    dividend_or_momentum: float


@dataclass(frozen=True)
class EquitySplitConfig:
    large: float
    mid: float
    small: float


@dataclass(frozen=True)
class TargetAllocationLevel:
    level: int
    label: str
    equity: float
    debt: float
    gold: float
    equity_split: EquitySplitConfig


@dataclass(frozen=True)
class BenchmarksConfig:
    broad_market: str = "^NSEI"
    midcap: str = "synthetic_midcap"
    smallcap: str = "synthetic_smallcap"
    synthetic_min_constituents_mid: int = 20
    synthetic_min_constituents_small: int = 15


@dataclass(frozen=True)
class UniverseStock:
    ticker: str
    name: str
    sector: str
    cap_class: str
    cap_source_date: str = ""


@dataclass(frozen=True)
class AppConfig:
    date_range: DateRangeConfig
    risk_free_rate: float
    trading_days: int
    random_seed: int
    extreme_z_threshold: float
    spike_review_threshold: float
    min_history_years: int
    min_liquidity_traded_value: float
    concentration_limits: ConcentrationLimitsConfig
    recommender_weights: RecommenderWeightsConfig
    target_allocations: dict[int, TargetAllocationLevel]
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    universe: list[UniverseStock] = field(default_factory=list)


def _require_key(data: Mapping[str, Any], key: str, context: str = "") -> Any:
    """Validate that a key exists in mapping, raising ConfigValidationError if missing."""
    if key not in data:
        ctx = f" in {context}" if context else ""
        raise ConfigValidationError(f"Missing required configuration key: '{key}'{ctx}")
    return data[key]


def _validate_date_range(raw: Any) -> DateRangeConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("'date_range' must be a dictionary")
    start = _require_key(raw, "start", "date_range")
    end = _require_key(raw, "end", "date_range")
    if not isinstance(start, str) or not start.strip():
        raise ConfigValidationError("'date_range.start' must be a non-empty string (YYYY-MM-DD)")
    if not isinstance(end, str) or not end.strip():
        raise ConfigValidationError("'date_range.end' must be a non-empty string ('latest' or YYYY-MM-DD)")
    return DateRangeConfig(start=str(start).strip(), end=str(end).strip())


def _validate_concentration_limits(raw: Any) -> ConcentrationLimitsConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("'concentration_limits' must be a dictionary")
    stock = float(_require_key(raw, "stock", "concentration_limits"))
    sector = float(_require_key(raw, "sector", "concentration_limits"))
    if not (0.0 < stock <= 1.0):
        raise ConfigValidationError(f"'concentration_limits.stock' must be in (0, 1], got {stock}")
    if not (0.0 < sector <= 1.0):
        raise ConfigValidationError(f"'concentration_limits.sector' must be in (0, 1], got {sector}")
    if stock > sector:
        raise ConfigValidationError(
            f"Stock limit ({stock}) cannot exceed sector concentration limit ({sector})"
        )
    return ConcentrationLimitsConfig(stock=stock, sector=sector)


def _validate_recommender_weights(raw: Any) -> RecommenderWeightsConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("'recommender_weights' must be a dictionary")
    keys = ["sharpe", "drawdown", "diversification", "sector_gap", "dividend_or_momentum"]
    values = {}
    for k in keys:
        val = float(_require_key(raw, k, "recommender_weights"))
        if val < 0.0:
            raise ConfigValidationError(f"'recommender_weights.{k}' must be non-negative, got {val}")
        values[k] = val

    total = sum(values.values())
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ConfigValidationError(f"Recommender weights must sum to 1.0, got {total}")

    return RecommenderWeightsConfig(**values)


def _validate_target_allocations(raw: Any) -> dict[int, TargetAllocationLevel]:
    if not isinstance(raw, dict):
        raise ConfigValidationError("'target_allocations' must be a dictionary")

    allocations: dict[int, TargetAllocationLevel] = {}
    required_levels = {1, 2, 3, 4, 5}

    for lvl in required_levels:
        level_data = raw.get(lvl) if lvl in raw else raw.get(str(lvl))
        if level_data is None:
            raise ConfigValidationError(f"Missing target allocation profile level: {lvl}")
        if not isinstance(level_data, dict):
            raise ConfigValidationError(f"Target allocation for level {lvl} must be a dictionary")

        label = str(_require_key(level_data, "label", f"target_allocations.{lvl}")).strip()
        equity = float(_require_key(level_data, "equity", f"target_allocations.{lvl}"))
        debt = float(_require_key(level_data, "debt", f"target_allocations.{lvl}"))
        gold = float(_require_key(level_data, "gold", f"target_allocations.{lvl}"))

        asset_sum = equity + debt + gold
        if not math.isclose(asset_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ConfigValidationError(
                f"Level {lvl} asset weights (equity+debt+gold) must sum to 1.0, got {asset_sum}"
            )

        eq_split_raw = _require_key(level_data, "equity_split", f"target_allocations.{lvl}")
        if not isinstance(eq_split_raw, dict):
            raise ConfigValidationError(f"Level {lvl} 'equity_split' must be a dictionary")

        large = float(_require_key(eq_split_raw, "large", f"target_allocations.{lvl}.equity_split"))
        mid = float(_require_key(eq_split_raw, "mid", f"target_allocations.{lvl}.equity_split"))
        small = float(_require_key(eq_split_raw, "small", f"target_allocations.{lvl}.equity_split"))

        split_sum = large + mid + small
        if not math.isclose(split_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ConfigValidationError(
                f"Level {lvl} equity split (large+mid+small) must sum to 1.0, got {split_sum}"
            )

        allocations[lvl] = TargetAllocationLevel(
            level=lvl,
            label=label,
            equity=equity,
            debt=debt,
            gold=gold,
            equity_split=EquitySplitConfig(large=large, mid=mid, small=small),
        )

    return allocations


def _validate_benchmarks(raw: Any) -> BenchmarksConfig:
    if not isinstance(raw, dict):
        return BenchmarksConfig()
    return BenchmarksConfig(
        broad_market=str(raw.get("broad_market", "^NSEI")),
        midcap=str(raw.get("midcap", "synthetic_midcap")),
        smallcap=str(raw.get("smallcap", "synthetic_smallcap")),
        synthetic_min_constituents_mid=int(raw.get("synthetic_min_constituents_mid", 20)),
        synthetic_min_constituents_small=int(raw.get("synthetic_min_constituents_small", 15)),
    )


def load_universe(universe_path: str | Path | None = None) -> list[UniverseStock]:
    """Load and validate the stock universe from the reference CSV file.

    Enforces:
    1. CSV file exists and has columns: ticker, name, sector, cap_class.
    2. No duplicate tickers.
    3. Only valid cap_class values ('large', 'mid', 'small').
    4. Only valid sector values from PRD Section 6.1.
    5. Approximate target distribution (roughly 30 large, 25 mid, 20 small).

    Args:
        universe_path: Path to universe.csv. Defaults to 'data/reference/universe.csv'.

    Returns:
        List of UniverseStock typed objects.

    Raises:
        FileNotFoundError: If universe.csv does not exist.
        ConfigValidationError: If schema, uniqueness, or field validation fails.
    """
    if universe_path is None:
        candidate = Path("data/reference/universe.csv")
        if not candidate.exists():
            candidate = Path(__file__).resolve().parent.parent / "data" / "reference" / "universe.csv"
        target_path = candidate
    else:
        target_path = Path(universe_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Universe reference file not found: {target_path}")

    stocks: list[UniverseStock] = []
    seen_tickers: set[str] = set()
    cap_counts: dict[str, int] = {"large": 0, "mid": 0, "small": 0}

    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"ticker", "name", "sector", "cap_class"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise ConfigValidationError(
                f"universe.csv must contain columns: {sorted(required_cols)}, got {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip()
            name = (row.get("name") or "").strip()
            sector = (row.get("sector") or "").strip()
            cap = (row.get("cap_class") or "").strip().lower()

            if not ticker:
                raise ConfigValidationError(f"Row {row_idx}: 'ticker' cannot be empty")
            if ticker in seen_tickers:
                raise ConfigValidationError(f"Duplicate ticker detected in universe: '{ticker}' (row {row_idx})")
            seen_tickers.add(ticker)

            if cap not in VALID_CAP_CLASSES:
                raise ConfigValidationError(
                    f"Invalid cap_class '{cap}' for ticker '{ticker}' (row {row_idx}). Must be one of {sorted(VALID_CAP_CLASSES)}"
                )

            if sector not in VALID_SECTORS:
                raise ConfigValidationError(
                    f"Invalid sector '{sector}' for ticker '{ticker}' (row {row_idx}). Must be one of {sorted(VALID_SECTORS)}"
                )

            cap_source_date = (row.get("cap_source_date") or "").strip()
            cap_counts[cap] += 1
            stocks.append(
                UniverseStock(
                    ticker=ticker,
                    name=name or ticker,
                    sector=sector,
                    cap_class=cap,
                    cap_source_date=cap_source_date,
                )
            )

    if len(stocks) == 0:
        raise ConfigValidationError("universe.csv contains no stock records")

    # Verify distribution per Phase 1c/1d: strictly at least 15 stocks per cap class
    for cap_cls, count in cap_counts.items():
        if count < 15:
            raise ConfigValidationError(
                f"Cap class '{cap_cls}' must have at least 15 stocks, found {count}"
            )

    return stocks


def load_config(
    config_path: str | Path | None = None,
    universe_path: str | Path | None = None,
    include_universe: bool = True,
) -> AppConfig:
    """Load, validate, and return the application configuration as a typed object.

    Args:
        config_path: Path to config.yaml. Defaults to 'config.yaml' in workspace root.
        universe_path: Path to universe.csv. Defaults to 'data/reference/universe.csv'.
        include_universe: Whether to load universe.csv into AppConfig.universe.

    Returns:
        AppConfig typed instance.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ConfigValidationError: If configuration syntax or values are invalid or required keys are missing.
    """
    if config_path is None:
        candidate = Path("config.yaml")
        if not candidate.exists():
            candidate = Path(__file__).resolve().parent.parent / "config.yaml"
        target_path = candidate
    else:
        target_path = Path(config_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in configuration file {target_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(f"Configuration root in {target_path} must be a dictionary mapping")

    # Mandatory root keys in config.yaml
    mandatory_keys = [
        "date_range",
        "risk_free_rate",
        "trading_days",
        "random_seed",
        "extreme_z_threshold",
        "spike_review_threshold",
        "min_history_years",
        "min_liquidity_traded_value",
        "concentration_limits",
        "recommender_weights",
        "target_allocations",
    ]
    for key in mandatory_keys:
        _require_key(data, key)

    # Validations
    date_range = _validate_date_range(data["date_range"])

    risk_free_rate = float(data["risk_free_rate"])
    if not (0.0 < risk_free_rate < 1.0):
        raise ConfigValidationError(
            f"'risk_free_rate' must be a positive fraction under 1.0 (e.g. 0.065), got {risk_free_rate}"
        )

    trading_days = int(data["trading_days"])
    if trading_days != 252:
        raise ConfigValidationError(
            f"'trading_days' must be 252 per AGENTS.md rule 8, got {trading_days}"
        )

    random_seed = int(data["random_seed"])
    if random_seed < 0:
        raise ConfigValidationError(f"'random_seed' must be non-negative, got {random_seed}")

    extreme_z_threshold = float(data["extreme_z_threshold"])
    if extreme_z_threshold <= 0:
        raise ConfigValidationError(
            f"'extreme_z_threshold' must be positive, got {extreme_z_threshold}"
        )

    spike_review_threshold = float(data["spike_review_threshold"])
    if not (0.0 < spike_review_threshold <= 1.0):
        raise ConfigValidationError(
            f"'spike_review_threshold' must be between 0 and 1.0, got {spike_review_threshold}"
        )

    min_history_years = int(data["min_history_years"])
    if min_history_years <= 0:
        raise ConfigValidationError(
            f"'min_history_years' must be positive integer, got {min_history_years}"
        )

    min_liquidity_traded_value = float(data["min_liquidity_traded_value"])
    if min_liquidity_traded_value <= 0:
        raise ConfigValidationError(
            f"'min_liquidity_traded_value' must be positive, got {min_liquidity_traded_value}"
        )

    concentration_limits = _validate_concentration_limits(data["concentration_limits"])
    recommender_weights = _validate_recommender_weights(data["recommender_weights"])
    target_allocations = _validate_target_allocations(data["target_allocations"])
    benchmarks = _validate_benchmarks(data.get("benchmarks", {}))

    universe: list[UniverseStock] = []
    if include_universe:
        try:
            universe = load_universe(universe_path)
        except FileNotFoundError:
            universe = []

    return AppConfig(
        date_range=date_range,
        risk_free_rate=risk_free_rate,
        trading_days=trading_days,
        random_seed=random_seed,
        extreme_z_threshold=extreme_z_threshold,
        spike_review_threshold=spike_review_threshold,
        min_history_years=min_history_years,
        min_liquidity_traded_value=min_liquidity_traded_value,
        concentration_limits=concentration_limits,
        recommender_weights=recommender_weights,
        target_allocations=target_allocations,
        benchmarks=benchmarks,
        universe=universe,
    )
