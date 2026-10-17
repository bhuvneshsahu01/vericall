"""
tests/test_stats.py
-------------------
Unit tests verifying bootstrap confidence interval bounds and paired McNemar statistical test.
"""

import numpy as np
import pytest
from vericall.eval.stats import compute_bootstrap_ci, compute_mcnemar_test


def test_bootstrap_ci_perfect_and_zero_scores():
    # All 1s
    mean, low, high = compute_bootstrap_ci([1.0] * 100, num_resamples=1000)
    assert mean == 100.0
    assert low == 100.0
    assert high == 100.0

    # All 0s
    mean, low, high = compute_bootstrap_ci([0.0] * 100, num_resamples=1000)
    assert mean == 0.0
    assert low == 0.0
    assert high == 0.0


def test_bootstrap_ci_bounds():
    # 500 samples with 70% ones
    rng = np.random.default_rng(42)
    scores = list(rng.choice([0.0, 1.0], size=500, p=[0.3, 0.7]))
    mean, low, high = compute_bootstrap_ci(scores, num_resamples=2000, seed=42)

    assert 65.0 <= mean <= 75.0
    assert 0.0 <= low <= mean
    assert mean <= high <= 100.0


def test_mcnemar_test_identical_predictions():
    y_base = [1, 0, 1, 1, 0, 1, 0, 0] * 50
    y_imp = list(y_base)
    stat, p_val = compute_mcnemar_test(y_base, y_imp)
    assert stat == 0.0
    assert p_val == 1.0


def test_mcnemar_test_statistically_significant_improvement():
    # Base: 40% accuracy, Improved: 85% accuracy on 500 samples
    y_base = [1] * 200 + [0] * 300
    y_imp = [1] * 425 + [0] * 75
    stat, p_val = compute_mcnemar_test(y_base, y_imp)

    assert stat > 10.0
    assert p_val < 0.001
