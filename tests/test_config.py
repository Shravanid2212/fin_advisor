"""Tests for configuration loader and schema validation."""

import math
from pathlib import Path

import pytest
import yaml

from src.config import (
    VALID_CAP_CLASSES,
    VALID_SECTORS,
    AppConfig,
    ConcentrationLimitsConfig,
    ConfigValidationError,
    DateRangeConfig,
    EquitySplitConfig,
    RecommenderWeightsConfig,
    TargetAllocationLevel,
    UniverseStock,
    load_config,
    load_universe,
)


def test_load_default_config_success():
    """Verify that default config.yaml loads correctly into a typed AppConfig object."""
    config = load_config()

    assert isinstance(config, AppConfig)
    assert isinstance(config.date_range, DateRangeConfig)
    assert config.date_range.start == "2016-01-01"
    assert config.date_range.end == "latest"

    assert config.risk_free_rate == pytest.approx(0.065)
    assert config.trading_days == 252
    assert config.random_seed == 42
    assert config.extreme_z_threshold == pytest.approx(4.0)
    assert config.spike_review_threshold == pytest.approx(0.20)
    assert config.min_history_years == 3
    assert config.min_liquidity_traded_value == pytest.approx(10000000.0)

    # Concentration limits
    assert isinstance(config.concentration_limits, ConcentrationLimitsConfig)
    assert config.concentration_limits.stock == pytest.approx(0.25)
    assert config.concentration_limits.sector == pytest.approx(0.40)

    # Recommender weights
    assert isinstance(config.recommender_weights, RecommenderWeightsConfig)
    assert config.recommender_weights.sharpe == pytest.approx(0.35)
    assert config.recommender_weights.drawdown == pytest.approx(0.20)
    assert config.recommender_weights.diversification == pytest.approx(0.25)
    assert config.recommender_weights.sector_gap == pytest.approx(0.10)
    assert config.recommender_weights.dividend_or_momentum == pytest.approx(0.10)
    assert math.isclose(
        config.recommender_weights.sharpe
        + config.recommender_weights.drawdown
        + config.recommender_weights.diversification
        + config.recommender_weights.sector_gap
        + config.recommender_weights.dividend_or_momentum,
        1.0,
    )

    # Target allocations
    assert len(config.target_allocations) == 5
    for lvl in range(1, 6):
        alloc = config.target_allocations[lvl]
        assert isinstance(alloc, TargetAllocationLevel)
        assert alloc.level == lvl
        assert math.isclose(alloc.equity + alloc.debt + alloc.gold, 1.0)
        assert isinstance(alloc.equity_split, EquitySplitConfig)
        assert math.isclose(
            alloc.equity_split.large + alloc.equity_split.mid + alloc.equity_split.small, 1.0
        )

    # Profile level 1 (Conservative)
    lvl1 = config.target_allocations[1]
    assert lvl1.label == "Conservative"
    assert lvl1.equity == pytest.approx(0.20)
    assert lvl1.debt == pytest.approx(0.70)
    assert lvl1.gold == pytest.approx(0.10)
    assert lvl1.equity_split.large == pytest.approx(1.00)
    assert lvl1.equity_split.mid == pytest.approx(0.00)
    assert lvl1.equity_split.small == pytest.approx(0.00)

    # Profile level 5 (Aggressive)
    lvl5 = config.target_allocations[5]
    assert lvl5.label == "Aggressive"
    assert lvl5.equity == pytest.approx(0.85)
    assert lvl5.debt == pytest.approx(0.10)
    assert lvl5.gold == pytest.approx(0.05)
    assert lvl5.equity_split.large == pytest.approx(0.40)
    assert lvl5.equity_split.mid == pytest.approx(0.35)
    assert lvl5.equity_split.small == pytest.approx(0.25)

    # Benchmarks & Universe
    assert config.benchmarks.broad_market == "^NSEI"
    assert len(config.universe) == 80
    assert any(stock.ticker == "HDFCBANK.NS" for stock in config.universe)


def test_universe_schema_and_distribution():
    """Verify that universe.csv has no duplicate tickers, only valid cap_class and sectors,

    and at least 15 stocks per cap class.
    """
    stocks = load_universe("data/reference/universe.csv")
    assert len(stocks) == 80

    tickers = [s.ticker for s in stocks]
    assert len(tickers) == len(set(tickers)), "Universe contains duplicate tickers"

    cap_counts: dict[str, int] = {}
    for s in stocks:
        assert isinstance(s, UniverseStock)
        assert s.ticker.endswith(".NS"), f"Ticker {s.ticker} should have .NS suffix"
        assert s.cap_class in VALID_CAP_CLASSES, f"Invalid cap_class: {s.cap_class}"
        assert s.sector in VALID_SECTORS, f"Invalid sector: {s.sector}"
        cap_counts[s.cap_class] = cap_counts.get(s.cap_class, 0) + 1

    # Phase 1d: Strictly at least 15 stocks per cap class and no duplicates
    for cap, count in cap_counts.items():
        assert count >= 15, f"Cap class {cap} has fewer than 15 stocks: {count}"


def test_universe_duplicate_ticker_fails(tmp_path: Path):
    """Verify that duplicate tickers in universe.csv trigger ConfigValidationError."""
    bad_csv = tmp_path / "dup_universe.csv"
    bad_csv.write_text(
        "ticker,name,sector,cap_class\n"
        "TCS.NS,TCS,IT,large\n"
        "TCS.NS,TCS Duplicate,IT,large\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="Duplicate ticker detected"):
        load_universe(bad_csv)


def test_universe_invalid_cap_class_fails(tmp_path: Path):
    """Verify that invalid cap_class values trigger ConfigValidationError."""
    bad_csv = tmp_path / "bad_cap.csv"
    bad_csv.write_text(
        "ticker,name,sector,cap_class\n"
        "TCS.NS,TCS,IT,mega_cap\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="Invalid cap_class"):
        load_universe(bad_csv)


def test_universe_invalid_sector_fails(tmp_path: Path):
    """Verify that invalid sector values trigger ConfigValidationError."""
    bad_csv = tmp_path / "bad_sec.csv"
    bad_csv.write_text(
        "ticker,name,sector,cap_class\n"
        "TCS.NS,TCS,Crypto & Web3,large\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="Invalid sector"):
        load_universe(bad_csv)


def test_universe_cap_class_under_15_fails(tmp_path: Path):
    """Verify that any cap class with fewer than 15 stocks raises ConfigValidationError."""
    csv_lines = ["ticker,name,sector,cap_class"]
    # 20 large
    for i in range(20):
        csv_lines.append(f"L{i}.NS,Large Company {i},IT,large")
    # 20 mid
    for i in range(20):
        csv_lines.append(f"M{i}.NS,Mid Company {i},IT,mid")
    # 14 small (old state before approved additions)
    for i in range(14):
        csv_lines.append(f"S{i}.NS,Small Company {i},IT,small")

    csv_path = tmp_path / "under_15_universe.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Cap class 'small' must have at least 15 stocks, found 14"):
        load_universe(csv_path)


def test_missing_file_raises_not_found(tmp_path: Path):
    """Verify that a nonexistent file path raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(missing_file)


def test_invalid_yaml_syntax(tmp_path: Path):
    """Verify that broken YAML syntax raises ConfigValidationError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("date_range: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="Invalid YAML"):
        load_config(bad_yaml)


def test_missing_mandatory_keys_fails_loudly(tmp_path: Path):
    """Verify that omitting mandatory keys raises ConfigValidationError."""
    base_data = {
        "date_range": {"start": "2016-01-01", "end": "latest"},
        "risk_free_rate": 0.065,
        "trading_days": 252,
        "random_seed": 42,
        "extreme_z_threshold": 4.0,
        "spike_review_threshold": 0.20,
        "min_history_years": 3,
        "min_liquidity_traded_value": 10000000.0,
        "concentration_limits": {"stock": 0.25, "sector": 0.40},
        "recommender_weights": {
            "sharpe": 0.35,
            "drawdown": 0.20,
            "diversification": 0.25,
            "sector_gap": 0.10,
            "dividend_or_momentum": 0.10,
        },
        "target_allocations": {
            lvl: {
                "label": f"Level {lvl}",
                "equity": 0.5,
                "debt": 0.4,
                "gold": 0.1,
                "equity_split": {"large": 0.5, "mid": 0.3, "small": 0.2},
            }
            for lvl in range(1, 6)
        },
    }

    # Test removing each mandatory key one by one
    for key in list(base_data.keys()):
        modified = dict(base_data)
        del modified[key]
        cfg_file = tmp_path / f"missing_{key}.yaml"
        cfg_file.write_text(yaml.dump(modified), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match=f"Missing required configuration key: '{key}'"):
            load_config(cfg_file)


def test_invalid_trading_days(tmp_path: Path):
    """Verify that trading_days != 252 raises ConfigValidationError per rule 8."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["trading_days"] = 365
    cfg_file = tmp_path / "bad_trading_days.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="trading_days.*must be 252"):
        load_config(cfg_file)


def test_invalid_recommender_weights_sum(tmp_path: Path):
    """Verify that recommender weights not summing to 1.0 raises ConfigValidationError."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["recommender_weights"]["sharpe"] = 0.50  # Sum becomes 1.15
    cfg_file = tmp_path / "bad_weights.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="weights must sum to 1.0"):
        load_config(cfg_file)


def test_invalid_target_allocation_sum(tmp_path: Path):
    """Verify that allocation asset sums != 1.0 raise ConfigValidationError."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["target_allocations"][1]["equity"] = 0.50  # 0.50 + 0.70 + 0.10 != 1.0
    cfg_file = tmp_path / "bad_alloc.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="asset weights.*must sum to 1.0"):
        load_config(cfg_file)


def test_invalid_concentration_limits(tmp_path: Path):
    """Verify that stock limit exceeding sector limit raises ConfigValidationError."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["concentration_limits"] = {"stock": 0.50, "sector": 0.40}
    cfg_file = tmp_path / "bad_limits.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Stock limit.*cannot exceed sector"):
        load_config(cfg_file)
