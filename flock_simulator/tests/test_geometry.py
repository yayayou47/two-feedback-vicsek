"""Unit tests for flock_simulator.core.geometry."""
from __future__ import annotations

import numpy as np
import pytest

from flock_simulator.core.geometry import (
    build_cell_list, normalize_angle,
)


class TestNormalizeAngle:
    def test_zero(self):
        assert normalize_angle(0.0) == 0.0

    def test_inside(self):
        assert normalize_angle(1.0) == pytest.approx(1.0)
        assert normalize_angle(-1.0) == pytest.approx(-1.0)

    def test_pi_branch(self):
        # exactly +pi maps to -pi (half-open interval [-pi, pi))
        assert normalize_angle(np.pi) == pytest.approx(-np.pi, abs=1e-12)

    def test_wraps_2pi(self):
        for k in (-3, -2, -1, 1, 2, 3):
            theta = 0.7 + 2.0 * np.pi * k
            assert normalize_angle(theta) == pytest.approx(0.7, abs=1e-10)


class TestBuildCellList:
    def test_partition(self):
        # every particle should appear in exactly one cell
        rng = np.random.default_rng(0)
        L = 10.0
        n_cell = 5
        N = 200
        x = rng.uniform(0.0, L, size=N)
        y = rng.uniform(0.0, L, size=N)
        head, nxt = build_cell_list(x, y, L, n_cell)
        seen = []
        for c in range(n_cell * n_cell):
            k = head[c]
            while k != -1:
                seen.append(int(k))
                k = nxt[k]
        assert sorted(seen) == list(range(N))

    def test_cell_index_correct(self):
        # particle at (0.6, 0.4) in L=2, n_cell=2 -> cell (0, 0).
        x = np.array([0.6])
        y = np.array([0.4])
        head, nxt = build_cell_list(x, y, 2.0, 2)
        assert head[0 * 2 + 0] == 0

    def test_periodic_at_boundary(self):
        # particle exactly at L should wrap to 0 in the cell index
        # (because Python int(L/cell) % n_cell == 0).
        x = np.array([2.0 - 1e-15])
        y = np.array([0.5])
        head, nxt = build_cell_list(x, y, 2.0, 2)
        # cell row = int(1.999/1) = 1 % 2 = 1
        assert head[1 * 2 + 0] == 0
