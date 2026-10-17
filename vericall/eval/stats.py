"""
vericall/eval/stats.py
----------------------
Statistical verification utilities:
1. 95% Bootstrap Confidence Intervals (10,000 resamples)
2. Paired McNemar Test (statsmodels / scipy) for hypothesis testing
"""

from typing import List, Tuple
import numpy as np


def compute_bootstrap_ci(
    scores: List[float],
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes empirical mean and non-parametric bootstrap confidence interval.

    Returns:
    --------
    (mean_pct, ci_lower_pct, ci_upper_pct)
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(arr) * 100.0)
    rng = np.random.default_rng(seed)

    boot_means = np.empty(num_resamples, dtype=np.float64)
    for i in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample) * 100.0

    alpha = 1.0 - confidence_level
    lower_pct = float(np.percentile(boot_means, (alpha / 2.0) * 100.0))
    upper_pct = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100.0))

    return round(mean_val, 2), round(lower_pct, 2), round(upper_pct, 2)


def compute_mcnemar_test(
    y_baseline: List[int],
    y_improved: List[int],
) -> Tuple[float, float]:
    """
    Computes paired McNemar test statistic and p-value between two models.
    y_baseline: binary pass (1) or fail (0) predictions for baseline model.
    y_improved: binary pass (1) or fail (0) predictions for improved model.

    Contingency Table:
                     Improved Pass (1)   Improved Fail (0)
    Baseline Pass (1)       a                   b
    Baseline Fail (0)       c                   d

    b: Baseline passed, Improved failed (discordant)
    c: Baseline failed, Improved passed (discordant)
    """
    assert len(y_baseline) == len(y_improved), "Prediction lengths must match"

    b = sum(1 for base, imp in zip(y_baseline, y_improved) if base == 1 and imp == 0)
    c = sum(1 for base, imp in zip(y_baseline, y_improved) if base == 0 and imp == 1)

    total_discordant = b + c
    if total_discordant == 0:
        return 0.0, 1.0

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        # 2x2 table: [[a, b], [c, d]]
        a = sum(1 for base, imp in zip(y_baseline, y_improved) if base == 1 and imp == 1)
        d = sum(1 for base, imp in zip(y_baseline, y_improved) if base == 0 and imp == 0)
        table = [[a, b], [c, d]]
        result = mcnemar(table, exact=(total_discordant < 25), correction=True)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        # Standard continuity-corrected chi-square approximation
        from scipy import stats
        stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))
        return stat, p_val
