# PowerShell runner script for Financial Market Analytics
param (
    [Parameter(Position=0)]
    [ValidateSet("data", "test", "app", "help")]
    [string]$Command = "help"
)

switch ($Command) {
    "data" {
        Write-Host "Running Data Pipeline..." -ForegroundColor Cyan
        python -m src.clean.clean_market_data
        python -m src.features.build_features
        python -m src.stats.build_stats_tables
        python -m src.ingest.dq_checks
    }
    "test" {
        Write-Host "Running Pytest Test Suite..." -ForegroundColor Cyan
        python -m pytest -v
    }
    "app" {
        Write-Host "Launching Streamlit Web App..." -ForegroundColor Cyan
        streamlit run app/Home.py
    }
    default {
        Write-Host "Usage: .\run.ps1 [data | test | app]" -ForegroundColor Yellow
    }
}
