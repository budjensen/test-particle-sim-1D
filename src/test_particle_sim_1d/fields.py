"""
fields.py

Defines analytic electric and magnetic field configurations for a
1D-in-space, 3D-in-velocity (1D3V) plasma simulation.

Each field function takes an array of particle positions `z` and returns
a NumPy array of shape (N, 3), representing the (Ex, Ey, Ez) or (Bx, By, Bz)
components at each position.

Functions
---------
E_uniform : Return a uniform electric field of specified magnitude and direction.
B_uniform : Return a uniform magnetic field of specified magnitude and direction.
B_mirror  : Return a magnetic mirror field with strength increasing ∝ z².

Examples
--------
>>> import numpy as np
>>> from test_particle_sim_1d import fields
>>> z = np.linspace(-0.1, 0.1, 5)
>>> fields.E_uniform(z, E0=1.0, direction="x")
array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.]])
>>> fields.B_mirror(z, B0=1.0, z_mirror=0.05)[:3]
array([[0., 0., 5.],
       [0., 0., 2.],
       [0., 0., 1.]])
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------
# Electric field definitions
# ----------------------------------------------------------------------


def E_uniform(z: np.ndarray, E0: float = 0.0, direction: str = "z") -> np.ndarray:
    """
    Uniform electric field of magnitude E0 along a specified axis.

    Parameters
    ----------
    z : np.ndarray
        Particle positions (ignored for uniform field)
    E0 : float, optional
        Field strength [V/m] (default 0)
    direction : {'x', 'y', 'z'}, optional
        Axis of field direction (default 'z')

    Returns
    -------
    np.ndarray
        Electric field array of shape (len(z), 3)
    """
    E = np.zeros((len(z), 3))
    idx = {"x": 0, "y": 1, "z": 2}[direction]
    E[:, idx] = E0
    return E


# ----------------------------------------------------------------------
# Magnetic field definitions
# ----------------------------------------------------------------------


def B_uniform(z: np.ndarray, B0: float = 0.0, direction: str = "z") -> np.ndarray:
    """
    Uniform magnetic field of magnitude B0 along a specified axis.

    Parameters
    ----------
    z : np.ndarray
        Particle positions (ignored)
    B0 : float
        Magnetic field strength [T]
    direction : {'x', 'y', 'z'}, optional
        Direction of the field vector

    Returns
    -------
    np.ndarray
        Magnetic field array of shape (len(z), 3)
    """
    B = np.zeros((len(z), 3))
    idx = {"x": 0, "y": 1, "z": 2}[direction]
    B[:, idx] = B0
    return B


def B_mirror(z: np.ndarray, B0: float = 1.0, z_mirror: float = 0.05) -> np.ndarray:
    """
    Simple magnetic mirror field that increases with z^2 near the ends.

    Parameters
    ----------
    z : np.ndarray
        Positions along z [m]
    B0 : float
        Base magnetic field [T]
    z_mirror : float
        Mirror length scale [m]

    Returns
    -------
    np.ndarray
        Magnetic field [T] along z
    """
    Bz = B0 * (1 + (z / z_mirror) ** 2)
    return np.column_stack((np.zeros_like(Bz), np.zeros_like(Bz), Bz))
