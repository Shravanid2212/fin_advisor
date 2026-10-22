"""Portfolio holdings data structures, CSV/manual parser, and input validation.

Source of truth: docs/PRD.md Section 12.
Adheres strictly to AGENTS.md Hard Rules:
- Rule 5: User portfolio lives in session memory only (never written to disk or logs).
- Rule 6: No magic numbers; validation rules enforce positive quantities and clean tickers.
- Never crashes on invalid rows; builds an explicit ValidationReport.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import UniverseStock, load_universe


@dataclass
class StockHolding:
    """Represents a single parsed equity holding."""

    ticker: str
    quantity: float
    avg_buy_price: float | None = None
    is_analysed: bool = True
    name: str = ""
    sector: str = ""
    cap_class: str = ""
    latest_price: float = 0.0
    current_value: float = 0.0
    gain_loss: float = 0.0
    gain_loss_pct: float = 0.0


@dataclass(frozen=True)
class FundBucket:
    """Represents a non-stock mutual fund or asset bucket allocation.

    Categories: 'large_cap', 'mid_small_cap', 'index', 'debt', 'gold', 'other'.
    """

    category: str
    amount: float


@dataclass
class ValidationReport:
    """Captures anomalies, warnings, and dropped invalid rows during parsing."""

    is_valid: bool = True
    issues: list[str] = field(default_factory=list)
    duplicates_merged: list[dict[str, Any]] = field(default_factory=list)
    unknown_tickers: list[str] = field(default_factory=list)
    bad_quantity_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_price_tickers: list[str] = field(default_factory=list)
    valid_holdings_count: int = 0


@dataclass
class PortfolioInput:
    """Container for stock holdings, fund buckets, and the parsing validation report."""

    holdings: list[StockHolding] = field(default_factory=list)
    funds: list[FundBucket] = field(default_factory=list)
    validation_report: ValidationReport = field(default_factory=ValidationReport)


def normalize_ticker(raw_ticker: str, universe_dict: dict[str, UniverseStock]) -> tuple[str, bool]:
    """Normalize a raw ticker string and check whether it belongs to the tracked universe.

    Normalisation rules:
    1. Strip whitespace and uppercase.
    2. If ticker does not end in '.NS' and adding '.NS' matches a universe stock, use that.
    3. Return (normalized_ticker, is_analysed).

    Args:
        raw_ticker: Raw input string (e.g. 'infy', 'TCS.NS', 'AAPL').
        universe_dict: Mapping of uppercase ticker to UniverseStock.

    Returns:
        Tuple of (clean_ticker, is_in_universe).
    """
    cleaned = str(raw_ticker).strip().upper()
    if not cleaned:
        return "", False

    if cleaned in universe_dict:
        return cleaned, True

    ns_candidate = f"{cleaned}.NS"
    if ns_candidate in universe_dict:
        return ns_candidate, True

    return cleaned, False


def parse_fund_buckets(fund_rows: list[dict[str, Any]] | None) -> tuple[list[FundBucket], list[str]]:
    """Parse and validate fund allocation buckets.

    Categories are mapped to standard tokens:
    'large_cap', 'mid_small_cap', 'index', 'debt', 'gold', 'other'.

    Args:
        fund_rows: List of dicts with 'category' and 'amount' keys.

    Returns:
        Tuple of (valid_funds, issues).
    """
    if not fund_rows:
        return [], []

    funds: list[FundBucket] = []
    issues: list[str] = []

    cat_map = {
        "large": "large_cap",
        "large-cap": "large_cap",
        "large_cap": "large_cap",
        "large cap": "large_cap",
        "mid": "mid_small_cap",
        "small": "mid_small_cap",
        "mid_small": "mid_small_cap",
        "mid-small": "mid_small_cap",
        "mid/small": "mid_small_cap",
        "mid/small-cap": "mid_small_cap",
        "mid_small_cap": "mid_small_cap",
        "index": "index",
        "equity_index": "index",
        "debt": "debt",
        "liquid": "debt",
        "fixed_income": "debt",
        "bond": "debt",
        "gold": "gold",
        "precious_metals": "gold",
        "other": "other",
    }

    for row in fund_rows:
        raw_cat = str(row.get("category", "")).strip().lower()
        cat = cat_map.get(raw_cat, "other")
        raw_amount = row.get("amount", 0.0)
        try:
            amount = float(raw_amount)
            if amount < 0 or math.isnan(amount) or math.isinf(amount):
                issues.append(f"Ignored fund bucket '{raw_cat}' with invalid non-positive amount: {raw_amount}")
                continue
            if amount > 0:
                funds.append(FundBucket(category=cat, amount=amount))
        except (ValueError, TypeError):
            issues.append(f"Ignored fund bucket '{raw_cat}' with non-numeric amount: {raw_amount}")

    return funds, issues


def parse_raw_stock_records(
    records: list[dict[str, Any]],
    universe: list[UniverseStock] | None = None,
    fund_rows: list[dict[str, Any]] | None = None,
) -> PortfolioInput:
    """Internal core parser for stock entries with duplicate merging and bad-row isolation.

    Args:
        records: List of raw row dicts containing ticker, quantity, and avg_buy_price.
        universe: Optional list of UniverseStock. Defaults to load_universe().
        fund_rows: Optional list of fund allocation rows.

    Returns:
        Validated PortfolioInput object.
    """
    if universe is None:
        universe = load_universe()

    universe_dict: dict[str, UniverseStock] = {stock.ticker.upper(): stock for stock in universe}

    report = ValidationReport()
    funds, fund_issues = parse_fund_buckets(fund_rows)
    report.issues.extend(fund_issues)

    # 1. First pass: Clean tickers and filter bad quantities
    intermediate_rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(records, start=1):
        # Flexible key extraction
        raw_ticker = row.get("ticker") or row.get("symbol") or row.get("stock") or ""
        raw_qty = row.get("quantity") or row.get("qty") or row.get("shares")
        raw_price = row.get("avg_buy_price") or row.get("avg_price") or row.get("buy_price") or row.get("price")

        # Skip empty row
        if not raw_ticker and raw_qty is None and raw_price is None:
            continue

        clean_ticker, is_analysed = normalize_ticker(raw_ticker, universe_dict)
        if not clean_ticker:
            report.issues.append(f"Row {row_idx}: Skipped entry with empty ticker symbol.")
            continue

        # Validate quantity
        try:
            if raw_qty is None:
                raise ValueError("Quantity is required")
            qty = float(raw_qty)
            if qty <= 0.0 or math.isnan(qty) or math.isinf(qty):
                raise ValueError(f"Quantity must be strictly positive, got {qty}")
        except (ValueError, TypeError) as e:
            report.bad_quantity_rows.append({"row_index": row_idx, "raw_row": row, "reason": str(e)})
            report.issues.append(f"Row {row_idx}: Dropped {raw_ticker} due to invalid quantity ({raw_qty}).")
            continue

        # Validate optional price
        buy_price: float | None = None
        if raw_price is not None and str(raw_price).strip() != "":
            try:
                p = float(raw_price)
                if p >= 0.0 and not math.isnan(p) and not math.isinf(p):
                    buy_price = p
                else:
                    report.issues.append(
                        f"Row {row_idx}: Ignored invalid avg_buy_price ({raw_price}) for {clean_ticker}; set to None."
                    )
            except (ValueError, TypeError):
                report.issues.append(
                    f"Row {row_idx}: Non-numeric avg_buy_price ({raw_price}) for {clean_ticker}; set to None."
                )

        if not is_analysed:
            if clean_ticker not in report.unknown_tickers:
                report.unknown_tickers.append(clean_ticker)
            report.issues.append(
                f"Holding {clean_ticker} is outside the tracked universe; marked as 'not analysed'."
            )

        intermediate_rows.append({
            "ticker": clean_ticker,
            "quantity": qty,
            "avg_buy_price": buy_price,
            "is_analysed": is_analysed,
        })

    # 2. Second pass: Merge duplicates
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in intermediate_rows:
        grouped.setdefault(r["ticker"], []).append(r)

    holdings: list[StockHolding] = []
    for ticker, rows in grouped.items():
        is_analysed = rows[0]["is_analysed"]
        stock_info = universe_dict.get(ticker)

        if len(rows) == 1:
            r = rows[0]
            holdings.append(
                StockHolding(
                    ticker=ticker,
                    quantity=r["quantity"],
                    avg_buy_price=r["avg_buy_price"],
                    is_analysed=is_analysed,
                    name=stock_info.name if stock_info else ticker,
                    sector=stock_info.sector if stock_info else "Unanalysed",
                    cap_class=stock_info.cap_class if stock_info else "unanalysed",
                )
            )
        else:
            total_qty = sum(r["quantity"] for r in rows)
            # Weighted average price across rows where price is specified
            total_spend = sum(r["quantity"] * r["avg_buy_price"] for r in rows if r["avg_buy_price"] is not None)
            qty_with_price = sum(r["quantity"] for r in rows if r["avg_buy_price"] is not None)

            merged_avg_price = (total_spend / qty_with_price) if qty_with_price > 0 else None

            report.duplicates_merged.append({
                "ticker": ticker,
                "occurrences": len(rows),
                "total_quantity": total_qty,
                "merged_avg_price": round(merged_avg_price, 4) if merged_avg_price is not None else None,
            })
            report.issues.append(
                f"Merged {len(rows)} duplicate entries for {ticker} into total quantity {total_qty} "
                f"(weighted avg price: {merged_avg_price if merged_avg_price is not None else 'N/A'})."
            )

            holdings.append(
                StockHolding(
                    ticker=ticker,
                    quantity=total_qty,
                    avg_buy_price=merged_avg_price,
                    is_analysed=is_analysed,
                    name=stock_info.name if stock_info else ticker,
                    sector=stock_info.sector if stock_info else "Unanalysed",
                    cap_class=stock_info.cap_class if stock_info else "unanalysed",
                )
            )

    report.valid_holdings_count = len(holdings)
    report.is_valid = len(report.bad_quantity_rows) == 0

    return PortfolioInput(holdings=holdings, funds=funds, validation_report=report)


def parse_manual_holdings(
    stock_rows: list[dict[str, Any]],
    fund_rows: list[dict[str, Any]] | None = None,
    universe: list[UniverseStock] | None = None,
) -> PortfolioInput:
    """Parse manual holdings entered via UI form.

    Args:
        stock_rows: List of dicts with 'ticker', 'quantity', and optional 'avg_buy_price'.
        fund_rows: Optional list of dicts with 'category' and 'amount'.
        universe: Optional UniverseStock reference list.

    Returns:
        PortfolioInput container.
    """
    return parse_raw_stock_records(stock_rows, universe=universe, fund_rows=fund_rows)


def parse_holdings_csv(
    csv_source: str | Path | io.StringIO,
    fund_rows: list[dict[str, Any]] | None = None,
    universe: list[UniverseStock] | None = None,
) -> PortfolioInput:
    """Parse holdings from CSV content, file path, or StringIO.

    Expected columns (case-insensitive):
    ticker, quantity, avg_buy_price

    If header is absent, defaults to positional: ticker, quantity, avg_buy_price.

    Args:
        csv_source: CSV file path, string containing CSV lines, or StringIO buffer.
        fund_rows: Optional list of fund bucket dicts.
        universe: Optional UniverseStock reference list.

    Returns:
        PortfolioInput container with full ValidationReport.
    """
    if isinstance(csv_source, Path):
        text_content = csv_source.read_text(encoding="utf-8")
    elif isinstance(csv_source, str):
        p = Path(csv_source)
        if p.exists() and p.is_file():
            text_content = p.read_text(encoding="utf-8")
        else:
            text_content = csv_source
    elif hasattr(csv_source, "read"):
        text_content = csv_source.read()
        if isinstance(text_content, bytes):
            text_content = text_content.decode("utf-8")
    else:
        raise TypeError(f"Unsupported csv_source type: {type(csv_source)}")

    reader = csv.reader(io.StringIO(text_content))
    rows = [r for r in reader if r and any(cell.strip() for cell in r)]

    if not rows:
        return PortfolioInput(
            holdings=[],
            funds=[],
            validation_report=ValidationReport(is_valid=True, issues=["CSV source is empty"]),
        )

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(h in ["ticker", "symbol", "stock", "quantity", "qty"] for h in header)

    records: list[dict[str, Any]] = []

    if has_header:
        # Map known column positions
        ticker_idx = next((i for i, h in enumerate(header) if h in ["ticker", "symbol", "stock"]), 0)
        qty_idx = next((i for i, h in enumerate(header) if h in ["quantity", "qty", "shares"]), 1)
        price_idx = next(
            (i for i, h in enumerate(header) if h in ["avg_buy_price", "avg_price", "buy_price", "price"]),
            None,
        )

        for data_row in rows[1:]:
            t = data_row[ticker_idx].strip() if ticker_idx < len(data_row) else ""
            q = data_row[qty_idx].strip() if qty_idx < len(data_row) else None
            p = data_row[price_idx].strip() if price_idx is not None and price_idx < len(data_row) else None
            records.append({"ticker": t, "quantity": q, "avg_buy_price": p})
    else:
        # Assume positional: col 0 = ticker, col 1 = quantity, col 2 = avg_buy_price
        for data_row in rows:
            t = data_row[0].strip() if len(data_row) > 0 else ""
            q = data_row[1].strip() if len(data_row) > 1 else None
            p = data_row[2].strip() if len(data_row) > 2 else None
            records.append({"ticker": t, "quantity": q, "avg_buy_price": p})

    return parse_raw_stock_records(records, universe=universe, fund_rows=fund_rows)
