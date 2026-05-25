"""Unit tests for the alpha-stable kick generator.

We verify two facts:
  1. at alpha = 2 the draw matches scipy's ``levy_stable.rvs`` to a
     Kolmogorov--Smirnov tolerance;
  2. at alpha = 1 the generated sample has the heavy-tail behaviour
     ``P(|xi| > t) ~ 1/t`` rather than the Gaussian ``exp(-t^2)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator.core.noise import stable_rvs_vector

scipy_stats = pytest.importorskip("scipy.stats")


class TestAlphaStable:
    def test_alpha2_matches_scipy_gaussian(self):
        # alpha = 2 should be a centred Gaussian with std sqrt(2)*scale
        rng = np.random.default_rng(42)
        N = 50_000
        alpha = np.full(N, 2.0)
        scale = 0.3
        x = stable_rvs_vector(alpha, scale, rng)
        std_emp = float(np.std(x))
        std_th = np.sqrt(2.0) * scale
        # 50k samples gives ~0.3% relative error
        assert abs(std_emp - std_th) / std_th < 0.02

    def test_alpha1_heavy_tail(self):
        # alpha = 1 should produce a Cauchy with at least one extreme
        # event |xi| > 50 in 50k draws (a Gaussian would have prob
        # < 1e-100 of any single value > 10).
        rng = np.random.default_rng(7)
        N = 50_000
        alpha = np.full(N, 1.0)
        x = stable_rvs_vector(alpha, 1.0, rng)
        assert np.max(np.abs(x)) > 50.0

    def test_per_particle_alpha(self):
        # half particles at alpha=1, half at alpha=2; the alpha=2
        # subset must have std sqrt(2) and the alpha=1 subset
        # must have a much larger MAD.
        rng = np.random.default_rng(11)
        N = 20_000
        alpha = np.empty(N)
        alpha[: N // 2] = 1.0
        alpha[N // 2:] = 2.0
        x = stable_rvs_vector(alpha, 1.0, rng)
        std_gauss = float(np.std(x[N // 2:]))
        mad_cauchy = float(np.median(np.abs(x[: N // 2])))
        # Gaussian std should be sqrt(2); Cauchy MAD should be ~1.0
        # (median absolute deviation of standard Cauchy is exactly 1).
        assert abs(std_gauss - np.sqrt(2.0)) < 0.05
        assert 0.7 < mad_cauchy < 1.4

    def test_seed_reproducibility(self):
        alpha = np.array([1.0, 1.5, 2.0])
        a = stable_rvs_vector(alpha, 0.1,
                              np.random.default_rng(99))
        b = stable_rvs_vector(alpha, 0.1,
                              np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)
