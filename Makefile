# Makefile for Financial Market Analytics & Decision Support System

.PHONY: data test app help

help:
	@echo "Financial Market Analytics CLI"
	@echo "  make data   - Run data cleaning, feature engineering, and statistical tables"
	@echo "  make test   - Run automated test suite with pytest"
	@echo "  make app    - Launch Streamlit web application"

data:
	python -m src.clean.clean_market_data
	python -m src.features.build_features
	python -m src.stats.build_stats_tables
	python -m src.ingest.dq_checks

test:
	python -m pytest -v

app:
	streamlit run app/Home.py
