# tests/test_species.py
from __future__ import annotations

import numpy as np

from test_particle_sim_1d.particles import Species


def test_species_initialization():
    """Test that a species object can be created and is initially empty"""
    s = Species(q=-1.602e-19, m=9.109e-31, name="electron", capacity=10)

    # Basic sanity checks
    assert isinstance(s, Species)
    assert s.N == 0
    assert s.capacity == 10
    assert np.all(s.x == 0.0)
    assert np.all(s.vx == 0.0)


def test_add_particles_increases_count():
    """Test adding new particles updates arrays correctly"""
    s = Species(q=-1.602e-19, m=9.109e-31)
    x_init = np.array([0.1, 0.2, 0.3])
    vx_init = np.array([1.0, 1.0, 1.0])

    s.add_particles(x_init, vx_init)

    assert len(s) == 3
    np.testing.assert_allclose(s.x[:3], x_init)
    np.testing.assert_allclose(s.vx[:3], vx_init)
