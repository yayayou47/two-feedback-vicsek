"""Omnidirectional two-feedback Vicsek--Couzin flock simulator.

The public surface is intentionally narrow: the user instantiates
``FlockParams`` and feeds it to ``FlockSimulator``, then calls
``.step()`` and reads ``.polarisation()`` and
``.density_separation_index()`` (and the heavier observables in
:mod:`flock_simulator.observables`).

This package replaces the legacy
``version4/src/vicsek_double_adaptive.py`` simulator with three
substantive changes:

  - the rear blind cone is removed (omnidirectional vision);
  - the simulator is modular (geometry / interactions / noise /
    adaptivity / observables in separate files), which makes
    pytest coverage feasible;
  - the random generator is a NumPy ``Generator`` seeded via
    ``SeedSequence``, so reproducibility across architectures is
    guaranteed.

The legacy simulator remains available under
``version4/src/vicsek_double_adaptive.py`` for the validation
comparison; do not remove it until all manuscript figures have been
re-generated against the new model.
"""
from .simulator import FlockParams, FlockSimulator, FlockState

__all__ = ["FlockParams", "FlockSimulator", "FlockState"]
