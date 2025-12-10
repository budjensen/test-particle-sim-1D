"""
tests/test_exb_drift.py

Checks that a charged particle starting from rest in crossed E and B fields:
- develops the correct ExB drift velocity over time
- conserves gyration energy in the drift frame
"""

from __future__ import annotations

import numpy as np

from test_particle_sim_1d import fields
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species


def test_exb_drift_from_rest():
    q, m = 1.0, 1.0
    E0, B0 = 0.5, 1.0
    v_d_expected = -E0 / B0  # E (+x) x B (+z) = -ŷ

    # Create species (start at rest)
    sp = Species(q=q, m=m, capacity=1)
    sp.add_particles(np.array([0.0]), np.array([[0.0, 0.0, 0.0]]))

    # Define fields
    def E_func(z):
        return fields.E_uniform(z, E0=E0, direction="x")

    def B_func(z):
        return fields.B_uniform(z, B0=B0, direction="z")

    # Integration setup
    omega_c = q * B0 / m
    T = 2 * np.pi / omega_c
    dt = T / 200
    n_steps = int(20 * 2 * np.pi / (omega_c * dt))  # ~20 gyroperiods

    z, v = sp.z.copy(), np.column_stack((sp.vx, sp.vy, sp.vz))
    _, _, v_hist = boris_1d3v_z(
        z,
        v,
        np.array([q]),
        np.array([m]),
        E_func,
        B_func,
        dt,
        n_steps,
        record_history=True,
    )

    # Ignore transients (first few gyroperiods)
    n_transient = n_steps // 2
    v_hist_steady = v_hist[n_transient:, 0, :]

    # 1. Average drift velocity (in y)
    avg_vy = np.mean(v_hist_steady[:, 1])
    np.testing.assert_allclose(avg_vy, v_d_expected, rtol=1e-2)

    # 2. Check conservation of kinetic energy *in the drift frame*
    v_d = np.array([0.0, v_d_expected, 0.0])
    v_rel = v_hist_steady - v_d
    v_mag_rel = np.linalg.norm(v_rel, axis=1)
    np.testing.assert_allclose(v_mag_rel / v_mag_rel.mean(), 1.0, rtol=1e-3)
