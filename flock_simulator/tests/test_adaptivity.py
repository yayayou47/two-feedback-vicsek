"""Unit tests for the shared-sigmoid adaptivity layer."""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator.core.adaptivity import shared_sigmoid_fields


class TestSharedSigmoid:
    def test_isolated_particle_is_fast_and_cauchy(self):
        # n_ali = 0, n_star = 3 -> sigmoid value sigmoid(-6) ~ 0.0025
        # so v_i ~ v_max and alpha_i ~ alpha_min.
        v, a = shared_sigmoid_fields(
            np.zeros(5, dtype=np.int64),
            n_star=3.0, slope=2.0,
            v_min=0.005, v_max=0.05,
            alpha_min=1.0, alpha_max=2.0,
        )
        # sigmoid(-6) ~ 0.00247: tolerance must accept (alpha_max-alpha_min)*0.00247.
        assert v[0] == pytest.approx(0.05, abs=5e-3)
        assert a[0] == pytest.approx(1.0, abs=5e-3)

    def test_crowded_particle_is_slow_and_gaussian(self):
        v, a = shared_sigmoid_fields(
            np.full(5, 10, dtype=np.int64),
            n_star=3.0, slope=2.0,
            v_min=0.005, v_max=0.05,
            alpha_min=1.0, alpha_max=2.0,
        )
        # n_ali = 10, sigmoid(14) ~ 1 to within 1e-6.
        assert v[0] == pytest.approx(0.005, abs=1e-5)
        assert a[0] == pytest.approx(2.0, abs=1e-5)

    def test_freezing_motility(self):
        # v_min == v_max collapses to a constant speed regardless of n.
        v, a = shared_sigmoid_fields(
            np.array([0, 5, 50], dtype=np.int64),
            n_star=3.0, slope=2.0,
            v_min=0.05, v_max=0.05,
            alpha_min=1.0, alpha_max=2.0,
        )
        assert np.allclose(v, 0.05)

    def test_freezing_noise_shape(self):
        # alpha_min == alpha_max collapses to a constant alpha.
        v, a = shared_sigmoid_fields(
            np.array([0, 5, 50], dtype=np.int64),
            n_star=3.0, slope=2.0,
            v_min=0.005, v_max=0.05,
            alpha_min=1.0, alpha_max=1.0,
        )
        assert np.allclose(a, 1.0)

    def test_threshold_value(self):
        # at n = n_star, sigmoid is exactly 0.5 and v / alpha sit at
        # their midpoints.
        v, a = shared_sigmoid_fields(
            np.array([3], dtype=np.int64),
            n_star=3.0, slope=2.0,
            v_min=0.005, v_max=0.05,
            alpha_min=1.0, alpha_max=2.0,
        )
        assert v[0] == pytest.approx(0.5 * (0.005 + 0.05))
        assert a[0] == pytest.approx(1.5)
