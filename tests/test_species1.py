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
    assert np.all(s.z == 0.0)
    assert np.all(s.vx == 0.0)
    assert np.all(s.vy == 0.0)
    assert np.all(s.vz == 0.0)


def test_add_particles_increases_count():
    """Test adding new particles updates arrays correctly"""
    s = Species(q=-1.602e-19, m=9.109e-31)
    z_init = np.array([0.1, 0.2, 0.3])
    v_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    s.add_particles(z_init, v_init)

    assert len(s) == 3
    np.testing.assert_allclose(s.x[:3], z_init)
    np.testing.assert_allclose(s.vx[:3], v_init[:, 0])
    np.testing.assert_allclose(s.vy[:3], v_init[:, 1])
    np.testing.assert_allclose(s.vz[:3], v_init[:, 2])
