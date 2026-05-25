"""Unit tests for flock_simulator.core.interactions.

The key behavioural assertion of the omnidirectional kernel is that a
particle directly *behind* another (which the legacy blind-cone
kernel would have ignored) now contributes to the alignment count.
"""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator.core.geometry import build_cell_list
from flock_simulator.core.interactions import zonal_update


def _zonal(x, y, theta, L, R_r, R_a):
    n_cell = max(1, int(L / max(R_a, 1e-9)))
    head, nxt = build_cell_list(x, y, L, n_cell)
    return zonal_update(x, y, theta, L, R_r, R_a, head, nxt, n_cell)


class TestNeighbourCount:
    def test_isolated(self):
        x = np.array([5.0])
        y = np.array([5.0])
        theta = np.array([0.0])
        new_theta, n_ali = _zonal(x, y, theta, 10.0, 0.5, 0.7)
        assert n_ali[0] == 0
        assert new_theta[0] == pytest.approx(0.0)  # heading kept

    def test_omnidirectional_rear_neighbour(self):
        # particle 0 looks "right" (theta=0). Neighbour 1 is exactly
        # behind at distance 0.6 (in the alignment annulus). The
        # omnidirectional kernel must count it; a blind-cone kernel
        # would not.
        x = np.array([5.0, 4.4])
        y = np.array([5.0, 5.0])
        theta = np.array([0.0, 0.5])
        new_theta, n_ali = _zonal(x, y, theta, 10.0, 0.5, 0.7)
        assert n_ali[0] == 1
        # particle 0 should now align with the rear neighbour's heading
        assert new_theta[0] == pytest.approx(0.5, abs=1e-10)

    def test_repulsion_priority(self):
        # one neighbour in the repulsion zone overrides any number of
        # alignment neighbours.
        x = np.array([0.0, 0.3, 0.6])
        y = np.array([0.0, 0.0, 0.0])
        theta = np.array([0.0, 1.0, 1.0])
        # particle 1 at d=0.3 (repulsion), particle 2 at d=0.6 (alignment)
        new_theta, n_ali = _zonal(x, y, theta, 10.0, 0.5, 0.7)
        # particle 0 should turn AWAY from particle 1 (i.e. -x direction
        # because particle 1 is to its +x), so new_theta[0] ~ pi.
        assert abs(abs(new_theta[0]) - np.pi) < 1e-10

    def test_periodic_wrap(self):
        # the two particles are separated by L - 0.1 in raw coordinates
        # but only 0.1 across the periodic boundary.
        L = 10.0
        x = np.array([0.05, L - 0.05])
        y = np.array([5.0, 5.0])
        theta = np.array([0.0, 0.5])
        new_theta, n_ali = _zonal(x, y, theta, L, 0.5, 0.7)
        # they sit at distance 0.1 across the wrap, which is in the
        # repulsion zone; so each repels the other.
        assert n_ali[0] == 0  # repulsion not counted as alignment
        assert n_ali[1] == 0
