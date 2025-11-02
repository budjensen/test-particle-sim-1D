"""
tests/test_gyromotion.py

Tests gyromotion for multiple particles with different initial velocities
in a uniform magnetic field using fields.py, integrators.py, and particles.py.

Checks:
- |v| conserved for each particle (energy conservation)
- Each particle maintains correct cyclotron frequency
- Phase shifts correspond to initial velocity directions
"""

from __future__ import annotations

import numpy as np

from test_particle_sim_1d import fields
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species


def test_multiple_particle_gyromotion():
    """Test that several particles gyrate independently and conserve |v|."""

    # ------------------------------------------------------------
    # Physical constants
    # ------------------------------------------------------------
    q = np.array([1.0, 1.0, 1.0])  # all same charge [C]
    m = np.array([1.0, 1.0, 1.0])  # all same mass [kg]
    B0 = 1.0  # magnetic field [T]
    v_perp = 1.0

    # Cyclotron frequency and period (same for all)
    omega_c = q[0] * B0 / m[0]
    T_cyclotron = 2 * np.pi / omega_c

    # ------------------------------------------------------------
    # Initialize 3 particles with different initial phases
    # ------------------------------------------------------------
    # One along +x, one along +y, one at 45° in xy-plane
    v_init = np.array(
        [
            [v_perp, 0.0, 0.0],  # particle 1
            [0.0, v_perp, 0.0],  # particle 2
            [v_perp / np.sqrt(2), v_perp / np.sqrt(2), 0.0],  # particle 3
        ]
    )
    z_init = np.zeros(3)

    sp = Species(q=1.0, m=1.0, name="multi", capacity=3)
    sp.add_particles(z_init, v_init)

    # ------------------------------------------------------------
    # Field setup
    # ------------------------------------------------------------
    def E_func(z):
        return fields.E_uniform(z, E0=0.0, direction="z")

    def B_func(z):
        return fields.B_uniform(z, B0=B0, direction="z")

    # ------------------------------------------------------------
    # Integrate one full cyclotron period
    # ------------------------------------------------------------
    dt = T_cyclotron / 400
    n_steps = 400

    z = sp.z.copy()
    v = np.column_stack((sp.vx, sp.vy, sp.vz))

    z_final, v_final = boris_1d3v_z(z, v, q, m, E_func, B_func, dt, n_steps)

    # ------------------------------------------------------------
    # Check 1: |v| conserved for all particles
    # ------------------------------------------------------------
    v_mag_initial = np.linalg.norm(v, axis=1)
    v_mag_final = np.linalg.norm(v_final, axis=1)
    np.testing.assert_allclose(v_mag_final, v_mag_initial, rtol=1e-6)

    # ------------------------------------------------------------
    # Check 2: Each particle returns to its original velocity direction
    # ------------------------------------------------------------
    np.testing.assert_allclose(v_final[:, :2], v[:, :2], atol=1e-3)

    # ------------------------------------------------------------
    # Check 3: z displacement small (1D motion assumption)
    # ------------------------------------------------------------
    np.testing.assert_allclose(z_final, z, atol=1e-6)
