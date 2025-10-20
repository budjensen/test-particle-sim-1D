"""
test_gyromotion.py

Test that the Boris 1D3V integrator produces correct gyromotion in a uniform B field.

Physical expectation:
---------------------
In a uniform magnetic field (B = Bz * e_z) and zero electric field,
a charged particle with initial velocity perpendicular to B
undergoes circular motion with constant |v|.

The cyclotron frequency is:
    ω_c = |q| * B / m

After one cyclotron period:
    T = 2π / ω_c
the velocity vector should return to its initial direction (within numerical tolerance).
"""

from __future__ import annotations

import numpy as np

from test_particle_sim_1d.integrators import boris_1d3v_z


def test_gyromotion_conservation():
    # --- Physical constants ---
    q = np.array([-1.602e-19])  # electron charge [C]
    m = np.array([9.109e-31])  # electron mass [kg]
    B0 = 1.0  # magnetic field [T]
    E0 = 0.0  # no electric field

    # --- Initial conditions ---
    z = np.zeros(1)
    v0 = np.array([[1e6, 0.0, 0.0]])  # velocity purely in x (perpendicular to Bz)

    # --- Field definitions ---
    def E_field(z):
        return np.stack(
            [np.full_like(z, E0), np.full_like(z, 0.0), np.full_like(z, 0.0)], axis=1
        )

    def B_field(z):
        return np.stack(
            [np.full_like(z, 0.0), np.full_like(z, 0.0), np.full_like(z, B0)], axis=1
        )

    # --- Cyclotron frequency and time step ---
    omega_c = np.abs(q[0]) * B0 / m[0]  # cyclotron frequency (rad/s)
    T_c = 2 * np.pi / omega_c  # cyclotron period (s)
    n_steps = 1000  # number of integration steps to use for one full gyro orbit
    dt = T_c / n_steps  # time step so that one period is divided into n_steps steps

    # --- Run one full gyro orbit ---
    _z_final, v_final = boris_1d3v_z(z, v0.copy(), q, m, E_field, B_field, dt, n_steps)

    # --- Expected: velocity magnitude conserved ---
    v_mag_initial = np.linalg.norm(v0)
    v_mag_final = np.linalg.norm(v_final)
    np.testing.assert_allclose(v_mag_final, v_mag_initial, rtol=1e-6)

    # --- Expected: velocity returns to (vx, vy) ≈ (v0, 0) after one full period ---
    np.testing.assert_allclose(
        v_final[0, 0], v0[0, 0], rtol=1e-4
    )  # vx close to initial
    np.testing.assert_allclose(v_final[0, 1], 0.0, atol=1e-4 * v_mag_initial)  # vy ≈ 0
    np.testing.assert_allclose(v_final[0, 2], 0.0, atol=1e-12)  # vz stays constant


if __name__ == "__main__":
    test_gyromotion_conservation()
