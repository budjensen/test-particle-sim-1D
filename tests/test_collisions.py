"""
tests/test_collisions.py

Unit tests for collision-frequency helpers in `initialization.py`:
- `get_max_collision_frequency`
- `validate_collision_frequency_timestep`

These tests check basic numeric correctness and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from test_particle_sim_1d.initialization import (
    get_max_collision_frequency,
    validate_collision_frequency_timestep,
)
from test_particle_sim_1d.particles import Species


def test_get_max_collision_frequency_scales_with_density():
    """Result should scale linearly with the maximum gas density."""
    energy = np.array([0.5])
    cross_section = np.array([0.3])
    gas_density = np.array([4.0])

    sp = Species(q=1.0, m=1.0)

    base = get_max_collision_frequency(energy, cross_section, gas_density, sp)
    doubled = get_max_collision_frequency(energy, cross_section, gas_density * 2.0, sp)

    assert doubled == pytest.approx(2.0 * base)


def test_validate_collision_frequency_timestep_negative_nu():
    """Negative collision frequency should raise ValueError."""
    with pytest.raises(
        ValueError, match=r"Collision frequency nu must be non-negative\."
    ):
        validate_collision_frequency_timestep(dt=1e-3, nu=-1.0)


def test_validate_collision_frequency_timestep_nonpositive_dt():
    """Non-positive dt should raise ValueError."""
    with pytest.raises(ValueError, match=r"Timestep dt must be a positive number\."):
        validate_collision_frequency_timestep(dt=0.0, nu=10.0)


def test_validate_collision_frequency_timestep_accepts_small_probability():
    """Small collision probability (coll_n <= 0.1) should not raise."""
    # coll_n = nu * dt = 10 * 1e-3 = 0.01 -> total prob ~0.00995 < 0.1
    validate_collision_frequency_timestep(dt=1e-3, nu=10.0)


def test_validate_collision_frequency_timestep_rejects_large_probability():
    """Large collision probability should raise ValueError."""
    # coll_n = 200 * 1e-3 = 0.2 -> total prob > 0.1
    with pytest.raises(ValueError, match=r"exceeds 0\.1\."):
        validate_collision_frequency_timestep(dt=1e-3, nu=200.0)
