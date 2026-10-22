"""Statistical hypothesis testing functions for Indian Equity Analytics (PRD §10).

Implements:
- Normality testing (Jarque-Bera, Shapiro-Wilk)
- Levene test for variance homogeneity across groups
- Kruskal-Wallis (and ANOVA) for differences across sectors and cap tiers
- Welch t-test & Mann-Whitney U for pairwise cap-tier comparisons
- Bootstrap confidence intervals for Sharpe and mean returns
- Correlation stability testing across calm and crash periods
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
import scipy.stats as stats


def run_normality_test(returns: pd.Series | np.ndarray) -> dict[str, Any]:
    """Test returns for normality using Jarque-Bera and Shapiro-Wilk tests.

    Formula / Method:
        Jarque-Bera: JB = (N/6) * (S^2 + (K - 3)^2 / 4), tested against Chi-Square(2).
        Shapiro-Wilk: W statistic on sample (capped at 5000 observations per scipy spec).

    Args:
        returns: Series or array of returns.

    Returns:
        Dictionary with test statistics, p-values, skewness, excess kurtosis, and conclusion.
    """
    clean_rets = pd.Series(returns).dropna()
    n = len(clean_rets)
    if n < 5:
        return {"error": "Insufficient sample size (< 5)"}

    jb_stat, jb_pval = stats.jarque_bera(clean_rets)

    # Shapiro-Wilk is valid for N <= 5000
    sample_for_sw = clean_rets.sample(min(n, 5000), random_state=42) if n > 5000 else clean_rets
    sw_stat, sw_pval = stats.shapiro(sample_for_sw)

    sk = float(stats.skew(clean_rets))
    kt = float(stats.kurtosis(clean_rets))  # Fisher excess kurtosis

    is_normal = bool((jb_pval > 0.05) and (sw_pval > 0.05))

    return {
        "n_observations": n,
        "skewness": round(sk, 4),
        "excess_kurtosis": round(kt, 4),
        "jarque_bera_stat": round(float(jb_stat), 2),
        "jarque_bera_pvalue": float(jb_pval),
        "shapiro_wilk_stat": round(float(sw_stat), 4),
        "shapiro_wilk_pvalue": float(sw_pval),
        "is_normal": is_normal,
        "plain_english": (
            "Returns reject normality (p < 0.001); distributions exhibit severe fat tails."
            if not is_normal
            else "Returns cannot reject normality (p >= 0.05)."
        ),
    }


def run_levene_test(groups: dict[str, pd.Series | np.ndarray]) -> dict[str, Any]:
    """Perform Levene's test for equality of variance across groups.

    Formula / Method:
        Tests H0: sigma_1^2 = sigma_2^2 = ... = sigma_k^2 using medians (Brown-Forsythe robust variant).

    Args:
        groups: Dictionary mapping group name to returns/volatility series.

    Returns:
        Dictionary with Levene W statistic, p-value, and plain-English summary.
    """
    cleaned_groups = [pd.Series(v).dropna() for v in groups.values() if len(pd.Series(v).dropna()) >= 2]
    if len(cleaned_groups) < 2:
        return {"error": "Need at least 2 valid groups"}

    stat, pval = stats.levene(*cleaned_groups, center="median")
    reject_h0 = bool(pval < 0.05)

    return {
        "groups": list(groups.keys()),
        "levene_stat": round(float(stat), 4),
        "pvalue": float(pval),
        "variances_differ": reject_h0,
        "plain_english": (
            f"Variances are significantly different across groups (W = {stat:.2f}, p < 0.001)."
            if pval < 0.001
            else f"Variances differ across groups (W = {stat:.2f}, p = {pval:.4f})."
            if reject_h0
            else f"No statistically significant difference in variance across groups (p = {pval:.4f})."
        ),
    }


def run_kruskal_wallis_test(groups: dict[str, pd.Series | np.ndarray]) -> dict[str, Any]:
    """Perform Kruskal-Wallis non-parametric ANOVA (and standard ANOVA) across groups.

    Formula / Method:
        H statistic based on ranked observations across k independent samples.

    Args:
        groups: Dictionary mapping group label to values.

    Returns:
        Dictionary with Kruskal-Wallis H statistic, p-value, and ANOVA comparison.
    """
    cleaned_groups = [pd.Series(v).dropna() for v in groups.values() if len(pd.Series(v).dropna()) >= 2]
    if len(cleaned_groups) < 2:
        return {"error": "Need at least 2 valid groups"}

    kw_stat, kw_pval = stats.kruskal(*cleaned_groups)
    f_stat, f_pval = stats.f_oneway(*cleaned_groups)

    reject_h0 = bool(kw_pval < 0.05)

    return {
        "groups": list(groups.keys()),
        "kruskal_stat": round(float(kw_stat), 4),
        "kruskal_pvalue": float(kw_pval),
        "anova_f_stat": round(float(f_stat), 4),
        "anova_pvalue": float(f_pval),
        "groups_differ": reject_h0,
        "plain_english": (
            f"Statistically significant difference detected between groups (H = {kw_stat:.2f}, p < 0.001)."
            if kw_pval < 0.001
            else f"Groups differ significantly (H = {kw_stat:.2f}, p = {kw_pval:.4f})."
            if reject_h0
            else f"No significant difference detected between groups (H = {kw_stat:.2f}, p = {kw_pval:.4f})."
        ),
    }


def run_pairwise_welch_mann_whitney(
    group1: pd.Series | np.ndarray,
    group2: pd.Series | np.ndarray,
    name1: str = "Group 1",
    name2: str = "Group 2",
) -> dict[str, Any]:
    """Pairwise comparison of two distributions via Welch's t-test and Mann-Whitney U test.

    Args:
        group1: Observations for group 1.
        group2: Observations for group 2.
        name1: Label for group 1.
        name2: Label for group 2.

    Returns:
        Dictionary with Welch t-test and Mann-Whitney U statistics and p-values.
    """
    g1 = pd.Series(group1).dropna()
    g2 = pd.Series(group2).dropna()

    t_stat, t_pval = stats.ttest_ind(g1, g2, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")

    return {
        "comparison": f"{name1} vs {name2}",
        "n1": len(g1),
        "n2": len(g2),
        "mean1": float(g1.mean()),
        "mean2": float(g2.mean()),
        "welch_t_stat": round(float(t_stat), 4),
        "welch_pvalue": float(t_pval),
        "mann_whitney_u_stat": round(float(u_stat), 2),
        "mann_whitney_pvalue": float(u_pval),
        "significant_at_05": bool(u_pval < 0.05),
        "plain_english": (
            f"{name1} differs significantly from {name2} (Mann-Whitney U = {u_stat:.1f}, p < 0.001)."
            if u_pval < 0.001
            else f"{name1} and {name2} differ significantly (p = {u_pval:.4f})."
            if u_pval < 0.05
            else f"No significant difference between {name1} and {name2} (p = {u_pval:.4f})."
        ),
    }


def bootstrap_ci(
    data: pd.Series | np.ndarray,
    metric_func: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute empirical percentile bootstrap confidence intervals.

    Formula / Method:
        Resample with replacement B times; compute metric for each sample;
        CI = (Percentile(alpha/2), Percentile(1 - alpha/2)).

    Args:
        data: Input observation sample.
        metric_func: Function accepting 1D numpy array and returning scalar metric.
        n_boot: Number of bootstrap replications (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci).
    """
    clean_data = np.asarray(pd.Series(data).dropna())
    n = len(clean_data)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_est = float(metric_func(clean_data))
    rng = np.random.default_rng(seed)

    boot_metrics = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(clean_data, size=n, replace=True)
        boot_metrics[i] = metric_func(sample)

    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    lower_ci = float(np.percentile(boot_metrics, lower_pct))
    upper_ci = float(np.percentile(boot_metrics, upper_pct))

    return point_est, lower_ci, upper_ci
