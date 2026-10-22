"""Universal command runner for Financial Market Analytics & Decision Support System.

Usage:
    python run.py data   # Run data cleaning, feature engineering, and statistical tables
    python run.py test   # Run full automated test suite
    python run.py app    # Launch Streamlit web application
"""

from __future__ import annotations

import subprocess
import sys

PYTHON_EXE = sys.executable


def run_data_pipeline() -> int:
    """Execute the end-to-end reproducible data and feature pipeline."""
    commands = [
        [PYTHON_EXE, "-m", "src.clean.clean_market_data"],
        [PYTHON_EXE, "-m", "src.features.build_features"],
        [PYTHON_EXE, "-m", "src.stats.build_stats_tables"],
        [PYTHON_EXE, "-m", "src.funds.ingest"],
        [PYTHON_EXE, "-m", "src.ingest.dq_checks"],
    ]

    print("\n" + "=" * 65)
    print(">> EXECUTING DATA PREPARATION & STATISTICAL PIPELINE")
    print("=" * 65)

    for cmd in commands:
        cmd_str = " ".join(cmd)
        print(f"\n[+] Running: {cmd_str}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"\n[!] Pipeline failed on step: {cmd_str}")
            return res.returncode

    print("\n" + "=" * 65)
    print(">> DATA PIPELINE COMPLETED SUCCESSFULLY!")
    print("Database data/market.duckdb and parquet tables are up to date.")
    print("=" * 65)
    return 0


def run_tests() -> int:
    """Execute the automated test suite."""
    print("\n" + "=" * 65)
    print(">> RUNNING AUTOMATED TEST SUITE")
    print("=" * 65)
    cmd = [PYTHON_EXE, "-m", "pytest", "-v"]
    res = subprocess.run(cmd)
    return res.returncode


def run_app() -> int:
    """Launch the interactive Streamlit web application."""
    print("\n" + "=" * 65)
    print(">> LAUNCHING STREAMLIT APPLICATION")
    print("=" * 65)
    cmd = [PYTHON_EXE, "-m", "streamlit", "run", "app/Home.py"]
    res = subprocess.run(cmd)
    return res.returncode


def print_help() -> None:
    """Display CLI usage instructions."""
    print(
        """
Financial Market Analytics & Decision Support System CLI

Usage:
    python run.py <command>

Available Commands:
    data    Run data cleaning, feature engineering, and statistical tables
    test    Execute pytest automated test suite
    app     Launch interactive Streamlit web application
    help    Display this help message
"""
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    action = sys.argv[1].strip().lower()

    if action == "data":
        sys.exit(run_data_pipeline())
    elif action == "test":
        sys.exit(run_tests())
    elif action == "app":
        sys.exit(run_app())
    elif action in ("help", "-h", "--help"):
        print_help()
        sys.exit(0)
    else:
        print(f"Unknown command: '{action}'")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
