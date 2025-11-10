"""
tests/test_species1.py


Unit tests for the Species class defined in ``test_particle_sim_1d/particles.py``.


These tests verify correct initialization, particle addition, and array
management behavior for the Species container, which stores particle
positions, velocities, and related attributes in a Structure-of-Arrays (SoA)
layout for 1D-in-space, 3D-in-velocity (1D3V) plasma simulations.


Functions
---------
test_species_initialization :
   Ensure that a Species object can be created and initialized correctly.
test_add_particles_increases_count :
   Verify that adding new particles updates internal arrays and counts as expected.


Examples
--------
>>> from test_particle_sim_1d.particles import Species
>>> import numpy as np
>>> s = Species(q=-1.602e-19, m=9.109e-31, name="electron", capacity=10)
>>> s.initialize_uniform(n=3, z_min=-0.1, z_max=0.1, v0=0.0)
>>> len(s)
3
>>> s.add_particles(np.array([0.2]), np.array([[0.0, 0.0, 1.0]]))
>>> len(s)
4
"""

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
    np.testing.assert_allclose(s.z[:3], z_init)
    np.testing.assert_allclose(s.vx[:3], v_init[:, 0])
    np.testing.assert_allclose(s.vy[:3], v_init[:, 1])
    np.testing.assert_allclose(s.vz[:3], v_init[:, 2])
