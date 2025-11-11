"""
tests/test_thermal_velocity.py

Unit tests for validate_thermal_velocity using electron species at
1 eV, and 4 eV.
"""

from __future__ import annotations

import pytest

from test_particle_sim_1d.initialization import (
    constants,
    validate_thermal_velocity_grid,
)
from test_particle_sim_1d.particles import Species


def make_electron():
    """Create a simple electron species for testing."""
    return Species(q=-constants.q_e, m=constants.m_e, name="electron")


def test_validate_v_th_1eV():
    sp = make_electron()

    # choose dt and dz so that 1 eV thermal velocity is small enough
    dt = 1e-6
    dz = 1.2

    validate_thermal_velocity_grid(dt=dt, dz=dz, temperature={"eV": 1.0}, species=sp)


def test_validate_v_th_4eV():
    sp = make_electron()

    dt = 1e-6
    dz = 1.2

    with pytest.raises(ValueError, match=r"is too small for thermal velocity"):
        validate_thermal_velocity_grid(
            dt=dt, dz=dz, temperature={"eV": 4.0}, species=sp
        )
