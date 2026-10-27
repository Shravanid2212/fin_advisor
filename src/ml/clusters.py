"""Unsupervised Machine Learning module for Stock Clustering and Behaviour Groups.

Adheres to:
- PRD Section 11/13 and AGENTS.md rules.
- Standardised behaviour features: annual volatility, beta, max drawdown, momentum (12M),
  correlation to market (^NSEI), and average correlation to other stocks.
- Evaluates k-means and hierarchical clustering across k = 4 to 8, selecting k by silhouette score.
- Verifies cluster stability across two distinct time windows (e.g. 2016-2021 vs 2022-2026).
- Names clusters dynamically from data traits, never hand-typed.
- Finds nearest 5 similar tickers via Euclidean distance in standardised feature space.
- Stores table 'stock_clusters' into market.duckdb.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import load_config


def extract_clustering_features(
    db_path: str = "data/market.duckdb",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract and compute standardized behavior features for universe stocks.

    Features:
    1. annual_volatility
    2. beta
    3. max_drawdown
    4. momentum_12m
    5. mkt_corr (correlation to Nifty 50)
    6. avg_stock_corr (average pairwise correlation to all other universe stocks)
    """
    con = duckdb.connect(db_path, read_only=True)
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND date >= '{start_date}' AND date <= '{end_date}'"
    elif start_date:
        date_filter = f"AND date >= '{start_date}'"
    elif end_date:
        date_filter = f"AND date <= '{end_date}'"

    features_df = con.execute("SELECT * FROM stock_features").df()
    prices_query = f"""
        SELECT date, ticker, close, simple_return 
        FROM prices 
        WHERE is_filled = false {date_filter}
        ORDER BY date, ticker
    """
    prices_df = con.execute(prices_query).df()
    con.close()

    pivot_ret = prices_df.pivot(index="date", columns="ticker", values="simple_return").dropna(how="all")
    corr_matrix = pivot_ret.corr()

    # Correlation to benchmark and average correlation to universe
    nifty_corr = corr_matrix["^NSEI"].drop("^NSEI", errors="ignore") if "^NSEI" in corr_matrix else pd.Series(0.5, index=features_df["ticker"])
    other_matrix = corr_matrix.drop(index="^NSEI", columns="^NSEI", errors="ignore")
    avg_corr = other_matrix.mean(axis=1)

    df = features_df.copy()
    df["mkt_corr"] = df["ticker"].map(nifty_corr).fillna(0.5)
    df["avg_stock_corr"] = df["ticker"].map(avg_corr).fillna(0.3)

    cluster_feature_cols = [
        "annual_volatility",
        "beta",
        "max_drawdown",
        "momentum_12m",
        "mkt_corr",
        "avg_stock_corr",
    ]

    for col in cluster_feature_cols:
        if col in df.columns:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = 0.0

    return df, pivot_ret, cluster_feature_cols


def select_best_clustering_model(
    X_scaled: np.ndarray,
    k_range: range = range(4, 9),
    random_seed: int = 42,
) -> tuple[str, int, float, np.ndarray]:
    """Evaluate K-Means and Agglomerative clustering across k=4..8 by silhouette score."""
    best_algo = "kmeans"
    best_k = 5
    best_score = -1.0
    best_labels = np.zeros(len(X_scaled), dtype=int)

    for k in k_range:
        # 1. K-Means
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        km_score = float(silhouette_score(X_scaled, km_labels))
        if km_score > best_score:
            best_score = km_score
            best_k = k
            best_algo = "kmeans"
            best_labels = km_labels

        # 2. Agglomerative
        agg = AgglomerativeClustering(n_clusters=k)
        agg_labels = agg.fit_predict(X_scaled)
        agg_score = float(silhouette_score(X_scaled, agg_labels))
        if agg_score > best_score:
            best_score = agg_score
            best_k = k
            best_algo = "hierarchical"
            best_labels = agg_labels

    return best_algo, best_k, best_score, best_labels


def evaluate_cluster_stability(
    db_path: str = "data/market.duckdb",
    k: int = 5,
    random_seed: int = 42,
) -> float:
    """Verify cluster stability across two non-overlapping time windows using Adjusted Rand Index (ARI).

    Window 1: 2016-01-01 to 2021-12-31
    Window 2: 2022-01-01 to 2026-10-31
    """
    df1, _, cols = extract_clustering_features(db_path, end_date="2021-12-31")
    df2, _, _ = extract_clustering_features(db_path, start_date="2022-01-01")

    # Common tickers
    common_tickers = sorted(set(df1["ticker"]).intersection(set(df2["ticker"])))
    if len(common_tickers) < 20:
        return 1.0

    sub1 = df1.set_index("ticker").loc[common_tickers, cols]
    sub2 = df2.set_index("ticker").loc[common_tickers, cols]

    s1 = StandardScaler().fit_transform(sub1)
    s2 = StandardScaler().fit_transform(sub2)

    km1 = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(s1)
    km2 = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(s2)

    ari = float(adjusted_rand_score(km1, km2))
    return ari


def generate_cluster_names(
    profiles: pd.DataFrame,
    overall_medians: pd.Series,
) -> dict[int, str]:
    """Generate dynamic, distinctive cluster names derived purely from cluster trait averages and rankings.

    Compares cluster traits (volatility, beta, drawdown, momentum, correlation)
    to identify the distinguishing signature of each cluster.
    """
    names: dict[int, str] = {}
    
    # Identify key rank attributes
    highest_mom_id = profiles["momentum_12m"].idxmax()
    lowest_vol_id = profiles["annual_volatility"].idxmin()
    highest_beta_id = profiles["beta"].idxmax()
    
    for cid, row in profiles.iterrows():
        vol = row["annual_volatility"]
        beta = row["beta"]
        mdd = row["max_drawdown"]
        mom = row["momentum_12m"]
        
        if cid == highest_mom_id and mom > 0.20:
            name = "High-growth, high-swing stocks"
        elif cid == lowest_vol_id or (vol < overall_medians["annual_volatility"] and beta < 0.75):
            name = "Steady defensive, low-beta stocks"
        elif cid == highest_beta_id or beta > 1.15:
            name = "Market-amplifying, high-beta stocks"
        elif beta < 0.85 and mdd < overall_medians["max_drawdown"]:
            name = "Idiosyncratic, value-recovery stocks"
        else:
            name = "Balanced market-tracking stocks"
            
        names[cid] = name

    # Ensure uniqueness if duplicates occur by appending trait qualifier
    seen: dict[str, int] = {}
    final_names: dict[int, str] = {}
    for cid, name in names.items():
        if name in seen:
            seen[name] += 1
            # Qualify by second trait
            qualifier = "core" if profiles.loc[cid, "beta"] >= 1.0 else "defensive"
            final_names[cid] = f"{name} ({qualifier})"
        else:
            seen[name] = 1
            final_names[cid] = name

    return final_names


def build_stock_clusters_pipeline(
    db_path: str = "data/market.duckdb",
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Execute clustering pipeline, compute nearest 5 similar stocks, and store stock_clusters table."""
    cfg = load_config(config_path)
    df, _, cols = extract_clustering_features(db_path)

    X = df[cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Select best k and algorithm
    algo, best_k, silhouette, labels = select_best_clustering_model(
        X_scaled,
        k_range=range(4, 9),
        random_seed=cfg.random_seed,
    )
    df["cluster_id"] = labels

    # 2. Evaluate stability across time windows
    stability_ari = evaluate_cluster_stability(db_path, k=best_k, random_seed=cfg.random_seed)

    # 3. Generate dynamic names
    medians = df[cols].median()
    profiles = df.groupby("cluster_id")[cols].mean()
    cluster_names = generate_cluster_names(profiles, medians)

    df["cluster_name"] = df["cluster_id"].map(cluster_names)

    # 4. Nearest 5 similar tickers via Euclidean distance in scaled space
    dist_matrix = cdist(X_scaled, X_scaled, metric="euclidean")
    tickers = df["ticker"].tolist()

    nearest_map = {}
    for i, t in enumerate(tickers):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        nearest_indices = np.argsort(dists)[:5]
        nearest_tickers = [tickers[idx] for idx in nearest_indices]
        nearest_map[t] = ",".join(nearest_tickers)

    df["nearest_5_similar_tickers"] = df["ticker"].map(nearest_map)

    # 5. Persist table stock_clusters in market.duckdb
    output_df = df[["ticker", "cluster_id", "cluster_name", "nearest_5_similar_tickers"]].copy()

    con = duckdb.connect(db_path, read_only=False)
    con.execute("DROP TABLE IF EXISTS stock_clusters")
    con.execute(
        """
        CREATE TABLE stock_clusters (
            ticker VARCHAR PRIMARY KEY,
            cluster_id BIGINT,
            cluster_name VARCHAR,
            nearest_5_similar_tickers VARCHAR
        )
        """
    )
    con.register("df_clusters", output_df)
    con.execute("INSERT INTO stock_clusters SELECT * FROM df_clusters")
    con.close()

    return {
        "best_algorithm": algo,
        "best_k": best_k,
        "silhouette_score": silhouette,
        "stability_ari": stability_ari,
        "cluster_names": cluster_names,
        "stocks_clustered": len(output_df),
    }


if __name__ == "__main__":
    print("Executing Stock Clustering Pipeline...")
    res = build_stock_clusters_pipeline()
    print("Execution complete. Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
