"""Unit tests for flock_simulator.observables.*"""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator.observables.order import (
    binder_cumulant, polarisation, susceptibility,
)
from flock_simulator.observables.spatial import (
    dbscan_cluster_sizes, density_separation_index,
    pair_correlation_separation,
)
from flock_simulator.observables.temporal import (
    heading_autocorrelation,
)


class TestPolarisation:
    def test_aligned(self):
        theta = np.zeros(100)
        assert polarisation(theta) == pytest.approx(1.0)

    def test_anti_aligned_pairs(self):
        theta = np.concatenate([np.zeros(50), np.full(50, np.pi)])
        assert polarisation(theta) == pytest.approx(0.0, abs=1e-12)

    def test_uniform_random_close_to_zero(self):
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=10_000)
        # 10k particles uniform on the circle gives |<exp>|<~ 0.02
        assert polarisation(theta) < 0.05


class TestSusceptibility:
    def test_zero_variance(self):
        # Constant trace has zero susceptibility (up to FP noise).
        chi = susceptibility(np.full(50, 0.7), N=1000)
        assert chi == pytest.approx(0.0, abs=1e-9)

    def test_linear_scaling_with_N(self):
        # chi = N * Var(phi), so doubling N doubles chi for the
        # same trace.
        trace = np.array([0.1, 0.2, 0.3, 0.4])
        chi1 = susceptibility(trace, N=100)
        chi2 = susceptibility(trace, N=200)
        assert chi2 == pytest.approx(2.0 * chi1)


class TestBinderCumulant:
    def test_constant_trace(self):
        # All measurements equal -> <phi^4> = <phi^2>^2 -> U4 = 1 - 1/3 = 2/3.
        assert binder_cumulant(np.full(50, 0.5)) == pytest.approx(2.0 / 3.0)

    def test_zero_trace_returns_zero(self):
        # m2 == 0 short-circuits to 0 to avoid 0/0.
        assert binder_cumulant(np.zeros(50)) == 0.0

    def test_gaussian_limit(self):
        # For a zero-mean Gaussian, U4 -> 1 - 3/3 = 0 in the
        # limit of many samples.
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50_000)
        assert binder_cumulant(x) == pytest.approx(0.0, abs=0.05)


class TestDensitySeparation:
    def test_uniform(self):
        # Particles spread uniformly should give s_sep close to 1.
        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, 10.0, size=5_000)
        y = rng.uniform(0.0, 10.0, size=5_000)
        s = density_separation_index(x, y, L=10.0, n_bins=10)
        # for 5000 / 100 cells = 50 per cell on average, the
        # max/median ratio should stay below ~1.5.
        assert 0.9 < s < 1.6

    def test_clustered(self):
        # All particles in one corner -> max bin much heavier
        # than median.
        rng = np.random.default_rng(11)
        x = rng.uniform(0.0, 1.0, size=400)
        y = rng.uniform(0.0, 1.0, size=400)
        # spread a few decoys across the box
        x = np.concatenate([x, rng.uniform(0.0, 10.0, size=200)])
        y = np.concatenate([y, rng.uniform(0.0, 10.0, size=200)])
        s = density_separation_index(x, y, L=10.0, n_bins=10)
        assert s > 5.0

    def test_empty_returns_one(self):
        # No particles: by construction the function returns 1.
        x = np.array([])
        y = np.array([])
        s = density_separation_index(x, y, L=10.0, n_bins=10)
        assert s == 1.0


class TestPairCorrelationSeparation:
    def test_uniform_close_to_one(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 20.0, size=4000)
        y = rng.uniform(0.0, 20.0, size=4000)
        s = pair_correlation_separation(x, y, L=20.0, R=1.0)
        # a homogeneous fluid has a narrow local-density field
        assert 1.0 <= s < 2.0

    def test_clustered_above_one(self):
        rng = np.random.default_rng(3)
        # minority dense blob + majority sparse background, so the
        # median local density lands in the dilute phase and the
        # 95th percentile in the blob.
        xb = rng.uniform(0.0, 2.0, size=250)
        yb = rng.uniform(0.0, 2.0, size=250)
        xs = rng.uniform(0.0, 20.0, size=1500)
        ys = rng.uniform(0.0, 20.0, size=1500)
        x = np.concatenate([xb, xs])
        y = np.concatenate([yb, ys])
        s = pair_correlation_separation(x, y, L=20.0, R=1.0)
        assert s > 2.0

    def test_too_few_particles_returns_one(self):
        assert pair_correlation_separation(
            np.array([1.0]), np.array([1.0]), L=10.0) == 1.0


class TestDBSCANClusterSizes:
    def test_two_well_separated_blobs(self):
        rng = np.random.default_rng(5)
        # two tight blobs far apart in a large box
        b1x = rng.normal(5.0, 0.2, size=50)
        b1y = rng.normal(5.0, 0.2, size=50)
        b2x = rng.normal(25.0, 0.2, size=60)
        b2y = rng.normal(25.0, 0.2, size=60)
        x = np.concatenate([b1x, b2x])
        y = np.concatenate([b1y, b2y])
        sizes = dbscan_cluster_sizes(x, y, L=30.0, eps=1.0,
                                      min_samples=4)
        assert len(sizes) == 2
        assert sorted(sizes.tolist()) == [50, 60]

    def test_sparse_returns_empty(self):
        # particles far apart relative to eps -> all noise.
        x = np.array([0.0, 5.0, 10.0, 15.0])
        y = np.array([0.0, 5.0, 10.0, 15.0])
        sizes = dbscan_cluster_sizes(x, y, L=20.0, eps=0.5,
                                      min_samples=4)
        assert len(sizes) == 0

    def test_periodic_wrap_merges_blob(self):
        # a single blob straddling the periodic boundary should
        # be one cluster, not two.
        rng = np.random.default_rng(7)
        ang = rng.uniform(0, 2 * np.pi, size=40)
        r = rng.uniform(0, 0.3, size=40)
        x = (r * np.cos(ang)) % 10.0   # centred on the x=0 wrap
        y = 5.0 + r * np.sin(ang)
        sizes = dbscan_cluster_sizes(x, y, L=10.0, eps=0.5,
                                      min_samples=4)
        assert len(sizes) == 1
        assert sizes[0] == 40


class TestHeadingAutocorrelation:
    def test_tau_zero_unity(self):
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, size=(20, 50))
        c = heading_autocorrelation(theta, np.array([0]))
        assert c[0] == 1.0

    def test_perfectly_persistent(self):
        # Constant heading -> autocorrelation = 1 at every lag.
        theta = np.full((30, 100), 0.4)
        c = heading_autocorrelation(theta,
                                     np.array([1, 5, 10, 50]))
        np.testing.assert_allclose(c, 1.0, atol=1e-12)

    def test_orthogonal_kick(self):
        # heading rotated by pi/2 at every step -> cos(d) = 0.
        n_t = 50
        theta = (np.pi / 2) * np.arange(n_t, dtype=float)
        theta = np.tile(theta, (5, 1))
        c = heading_autocorrelation(theta, np.array([1, 2, 3, 4]))
        # cos(pi/2)=0, cos(pi)=-1, cos(3pi/2)=0, cos(2pi)=1
        np.testing.assert_allclose(c, np.array([0., -1., 0., 1.]),
                                    atol=1e-12)
