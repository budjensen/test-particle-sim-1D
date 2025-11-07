"""
integrators.py

Boris integrator for 1D spatial / 3D velocity (1D3V) particle motion.

Particles move along z but have full 3D velocity.
The electric and magnetic fields depend only on z.

This module advances particle positions and velocities according to the
Lorentz force law:

    m dv/dt = q (E + v x B)
    dz/dt = v_z

The Boris algorithm is a second-order, time-centered scheme that splits the
electric and magnetic effects into separate half steps:
- Half electric acceleration
- Magnetic rotation
- Second half electric acceleration

References
----------
J. P. Boris, "Relativistic Plasma Simulation—Optimization of a Hybrid Code",
Proceedings of the Fourth Conference on Numerical Simulation of Plasmas, 1970.

Qin, H. et al. (2013). "Why is Boris algorithm so good?"
"""

from __future__ import annotations

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
    record_history: bool = False,
):
    """
    Advance particles in 1D (z) with 3D velocity using the Boris pusher.

    This algorithm integrates the Lorentz force equation in a time-centered,
    energy-conserving way. It applies a half-step electric acceleration,
    a magnetic rotation, and a second half electric acceleration.

    Parameters
    ----------
    z : np.ndarray
        Particle positions along z-axis [m], shape (N,).
    v : np.ndarray
        Particle velocities [m/s], shape (N, 3).
    q : np.ndarray
        Particle charges [C], shape (N,).
    m : np.ndarray
        Particle masses [kg], shape (N,).
    E_func : callable
        Function that returns the electric field array for given z positions.
        Must have signature ``E_func(z) -> np.ndarray`` of shape (N, 3).
    B_func : callable
        Function that returns the magnetic field array for given z positions.
        Must have signature ``B_func(z) -> np.ndarray`` of shape (N, 3).
    dt : float
        Time step [s].
    n_steps : int
        Number of time steps to advance.
    record_history : bool, optional
        If True, store and return full velocity history as a 3D array of
        shape (n_steps, N, 3). Default is False.

    Returns
    -------
    tuple of np.ndarray
        - z : np.ndarray
            Final particle positions along z, shape (N,).
        - v : np.ndarray
            Final particle velocities, shape (N, 3).
        - v_hist : np.ndarray, optional
            Velocity history (n_steps, N, 3), only returned if
            ``record_history=True``.

    Raises
    ------
    ValueError
        If z or v arrays have inconsistent shapes.
    TypeError
        If E_func or B_func do not return arrays of shape (N, 3).

    Notes
    -----
    The Boris scheme is widely used in plasma and particle-in-cell simulations
    due to its excellent long-term energy conservation and numerical stability.
    It is exact for constant magnetic fields when E=0.

    """
    q_over_m = (q / m)[:, None]

    if record_history:
        v_hist = np.zeros((n_steps, len(v), 3), dtype=v.dtype)

    for i in range(n_steps):
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

        if record_history:
            v_hist[i] = v

    if record_history:
        return z, v, v_hist
    return z, v
