"""
integrators.py

Boris integrator for 1D spatial / 3D velocity (1D3V) particle motion.

Particles move along z but have full 3D velocity.
The electric and magnetic fields depend only on z.

This module advances particle positions and velocities according to the
Lorentz force law:

    m dv/dt = q (E + v × B)
    dz/dt = v_z

The Boris algorithm is a second-order, time-centered scheme that splits the
electric and magnetic effects into separate half steps:
- Half electric acceleration
- Magnetic rotation
- Second half electric acceleration

References:
    J. P. Boris, "Relativistic Plasma Simulation—Optimization of a Hybrid Code",
    Proceedings of the Fourth Conference on Numerical Simulation of Plasmas, 1970.
    Qin, H. et al. (2013). "Why is Boris algorithm so good?"
"""

import numpy as np


def boris_1d3v_z(
    z: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    m: np.ndarray,
    E_func,
    B_func,
    dt: float,
    n_steps: int,
):
    """
    Advance 1D-in-z, 3D-in-velocity particles using the Boris algorithm.

    Args:
        z (np.ndarray): Particle positions along z [m], shape (N,)
        v (np.ndarray): Particle velocities [m/s], shape (N, 3)
        q (np.ndarray): Charges [C], shape (N,)
        m (np.ndarray): Masses [kg], shape (N,)
        E_func (callable): E(z) -> (N, 3) electric field [V/m]
        B_func (callable): B(z) -> (N, 3) magnetic field [T]
        dt (float): Time step [s]
        n_steps (int): Number of steps

    Returns:
        tuple[np.ndarray, np.ndarray]: Final (z, v)
    """
    q_over_m = (q / m)[:, None]

    for step in range(n_steps):
        # 1. Fields at current positions
        E = E_func(z)
        B = B_func(z)

        # 2. Half-step electric acceleration
        v_minus = v + 0.5 * dt * q_over_m * E

        # 3. Magnetic rotation
        t = q_over_m * B * (0.5 * dt)
        t_mag2 = np.sum(t * t, axis=1, keepdims=True)
        s = 2 * t / (1 + t_mag2)

        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)

        # 4. Second half-step electric acceleration
        v = v_plus + 0.5 * dt * q_over_m * E

        # 5. Position update using vz
        z += v[:, 2] * dt

    return z, v
