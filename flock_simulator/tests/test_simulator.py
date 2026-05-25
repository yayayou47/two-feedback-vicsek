"""Integration tests for the FlockSimulator class.

Black-box invariants the simulator must satisfy regardless of the
internal implementation:

  - positions stay within ``[0, L)`` after any number of steps;
  - polarisation is in ``[0, 1]``;
  - same seed -> bit-identical trajectory on the same machine;
  - parameter validation rejects ill-posed configurations.
"""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator import FlockParams, FlockSimulator


class TestFlockSimulator:
    def test_positions_in_box(self):
        p = FlockParams(N=300, L=10.0, eta=0.1, seed=1)
        sim = FlockSimulator(p)
        for _ in range(50):
            sim.step()
        assert (sim.state.x >= 0.0).all() and (sim.state.x < p.L).all()
        assert (sim.state.y >= 0.0).all() and (sim.state.y < p.L).all()

    def test_polarisation_bounds(self):
        p = FlockParams(N=200, L=10.0, eta=0.1, seed=3)
        sim = FlockSimulator(p)
        for _ in range(30):
            sim.step()
            phi = sim.polarisation()
            assert 0.0 <= phi <= 1.0 + 1e-12

    def test_seed_reproducibility(self):
        p1 = FlockParams(N=200, L=10.0, eta=0.15, seed=42)
        p2 = FlockParams(N=200, L=10.0, eta=0.15, seed=42)
        s1 = FlockSimulator(p1)
        s2 = FlockSimulator(p2)
        for _ in range(40):
            s1.step()
            s2.step()
        np.testing.assert_array_equal(s1.state.x, s2.state.x)
        np.testing.assert_array_equal(s1.state.theta, s2.state.theta)

    def test_different_seeds_diverge(self):
        s1 = FlockSimulator(FlockParams(N=200, L=10.0, seed=1))
        s2 = FlockSimulator(FlockParams(N=200, L=10.0, seed=2))
        for _ in range(20):
            s1.step()
            s2.step()
        assert not np.allclose(s1.state.x, s2.state.x)

    def test_motility_freeze_keeps_constant_speed(self):
        p = FlockParams(N=200, L=10.0, v_min=0.05, v_max=0.05,
                        alpha_min=1.0, alpha_max=2.0, seed=5)
        sim = FlockSimulator(p)
        for _ in range(20):
            sim.step()
            assert np.allclose(sim.state.v_i, 0.05)

    def test_topological_alignment_runs(self):
        # Topological mode with k_NN=4 should advance without
        # errors and produce a valid trajectory.
        p = FlockParams(N=300, L=10.0, eta=0.1, seed=3,
                        alignment="topological", k_NN=4)
        sim = FlockSimulator(p)
        for _ in range(20):
            sim.step()
        assert (sim.state.x >= 0.0).all() and (sim.state.x < p.L).all()
        assert 0.0 <= sim.polarisation() <= 1.0 + 1e-12

    def test_topological_seed_reproducibility(self):
        p1 = FlockParams(N=200, L=10.0, seed=7,
                         alignment="topological", k_NN=4)
        p2 = FlockParams(N=200, L=10.0, seed=7,
                         alignment="topological", k_NN=4)
        s1, s2 = FlockSimulator(p1), FlockSimulator(p2)
        for _ in range(30):
            s1.step(); s2.step()
        np.testing.assert_array_equal(s1.state.x, s2.state.x)
        np.testing.assert_array_equal(s1.state.theta, s2.state.theta)

    def test_topological_differs_from_metric(self):
        # Same seed, same parameters, different alignment kernel ->
        # trajectories must diverge after a few steps (otherwise
        # the topological dispatch is silently no-op).
        p_m = FlockParams(N=200, L=10.0, seed=1, alignment="metric")
        p_t = FlockParams(N=200, L=10.0, seed=1,
                          alignment="topological", k_NN=4)
        s_m, s_t = FlockSimulator(p_m), FlockSimulator(p_t)
        for _ in range(30):
            s_m.step(); s_t.step()
        assert not np.allclose(s_m.state.x, s_t.state.x)

    def test_decoupled_sigmoid_branch(self):
        # Two distinct sigmoid thresholds activate the decoupled
        # branch in simulator.step (two separate calls to
        # shared_sigmoid_fields, one per channel).
        p = FlockParams(N=200, L=10.0, seed=4,
                        n_star_v=1.0, n_star_alpha=5.0,
                        slope_v=2.0, slope_alpha=2.0)
        sim = FlockSimulator(p)
        for _ in range(20):
            sim.step()
        # v_i should drop quickly (low threshold for motility)
        # while alpha_i should stay near alpha_min for most particles
        # (high threshold for noise-shape).
        assert sim.state.v_i.mean() < 0.5 * (p.v_max + p.v_min)
        # at L = 10 and N = 200 the typical n_i is ~0.75 * 2 = 1.5,
        # so alpha_i stays close to alpha_min = 1.
        assert sim.state.alpha_i.mean() < 1.5

    def test_density_separation_index_method(self):
        # Exercise the snapshot s_sep method directly.
        p = FlockParams(N=300, L=10.0, seed=2)
        sim = FlockSimulator(p)
        for _ in range(30):
            sim.step()
        s = sim.density_separation_index()
        assert 0.5 < s < 100.0

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(v_min=0.1, v_max=0.05))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(alpha_min=2.0, alpha_max=1.0))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(R_r=0.7, R_a=0.5))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(L=-1.0))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(alpha_min=-0.1, alpha_max=2.0))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(alignment="banana"))
        with pytest.raises(ValueError):
            FlockSimulator(FlockParams(alignment="topological", k_NN=0))
