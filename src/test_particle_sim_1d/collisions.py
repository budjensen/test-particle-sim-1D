"""
collisions.py

Implements Monte Carlo Collision (MCC) logic for a 1D3V particle simulation.

This module handles the probabilistic nature of particle-neutral collisions.
It calculates collision probabilities based on the particle's relative velocity
to the background gas and updates particle velocities using elastic scattering
dynamics.

The collision probability is determined by:
    P_coll = 1 - exp(-n_g * sigma(g) * g * dt)

Functions
---------
apply_elastic_collisions : Main function to check for and apply collisions.
scatter_isotropic_3d     : Updates velocities of colliding particles (Elastic).

"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import particles


def apply_elastic_collisions(
    species: particles.Species,
    neutral_density: float,
    neutral_temp_K: float,
    neutral_mass: float,
    cross_section_func: Callable[[np.ndarray], np.ndarray],
    dt: float,
    seed: int | None = None,
) -> int:
    """
    Check for collisions between test particles and a background gas, then
    update velocities using elastic scattering.

    Parameters
    ----------
    species : Species
        The particle species object containing position and velocity arrays.
    neutral_density : float
        Number density of the background gas [m^-3].
    neutral_temp_K : float
        Temperature of the background gas in Kelvin [K].
    neutral_mass : float
        Mass of a single background gas particle [kg].
    cross_section_func : Callable
        A function that takes relative velocity array `g` [m/s] and returns
        cross-section array `sigma` [m^2].
    dt : float
        Time step [s].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    int
        The total number of collisions that occurred in this step.
    """
    rng = np.random.default_rng(seed)

    # 1. Access active particle data
    # We only care about the first N particles that are alive
    N = species.N
    if N == 0:
        return 0

    vx = species.vx[:N]
    vy = species.vy[:N]
    vz = species.vz[:N]

    # 2. Calculate EXACT relative velocity 'g'

    # Sample candidate neutrals for all active particles
    v_neutral_all = particles.sample_maxwellian(
        n=N,
        mass=neutral_mass,
        temperature={"K": neutral_temp_K},
        mean_velocity=0.0,
        seed=seed,
    )

    # Calculate vector relative velocity for every particle
    v_rel_x = vx - v_neutral_all[:, 0]
    v_rel_y = vy - v_neutral_all[:, 1]
    v_rel_z = vz - v_neutral_all[:, 2]

    # Compute magnitude g
    g = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

    # 3. Calculate Cross Section sigma(g)
    sigma = cross_section_func(g)

    # 4. Calculate Collision Probability (Eq. 2 in proposal)
    # P = 1 - exp(-n * sigma * g * dt)
    exponent = -neutral_density * sigma * g * dt
    P_coll = 1.0 - np.exp(exponent)

    # 5. Determine which particles collide (Vectorized Monte Carlo)
    # Generate random numbers [0, 1) for every particle
    R = rng.random(N)
    collision_mask = P_coll > R
    n_collisions = np.count_nonzero(collision_mask)

    if n_collisions == 0:
        return 0

    # 6. Resolve Collisions (Update Velocities)
    # Extract indices of colliding particles
    idx = np.where(collision_mask)[0]

    # Get velocities of colliding ions
    v_ion = np.column_stack((vx[idx], vy[idx], vz[idx]))

    # Reuse the specific neutrals that we calculated the collision probability against
    v_neutral = v_neutral_all[idx]

    # Perform Isotropic Elastic Scattering
    v_ion_new = scatter_isotropic_3d(
        v1=v_ion, v2=v_neutral, m1=species.m, m2=neutral_mass, rng=rng
    )

    # 7. Write new velocities back to the species object
    species.vx[idx] = v_ion_new[:, 0]
    species.vy[idx] = v_ion_new[:, 1]
    species.vz[idx] = v_ion_new[:, 2]

    return n_collisions


def scatter_isotropic_3d(
    v1: np.ndarray, v2: np.ndarray, m1: float, m2: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Perform elastic hard-sphere scattering in 3D.

    Updates the velocity of species 1 (v1) assuming isotropic scattering
    in the Center of Mass (CoM) frame.

    Parameters
    ----------
    v1 : np.ndarray
        Velocities of scattering particles (N, 3).
    v2 : np.ndarray
        Velocities of target particles (N, 3).
    m1 : float
        Mass of scattering particles.
    m2 : float
        Mass of target particles.
    rng : np.random.Generator
        Random number generator instance.

    Returns
    -------
    np.ndarray
        New velocities for species 1, shape (N, 3).
    """
    n_cols = len(v1)
    total_mass = m1 + m2

    # 1. Calculate Center of Mass Velocity
    v_cm = (m1 * v1 + m2 * v2) / total_mass

    # 2. Calculate Relative Velocity (v_rel = v1 - v2)
    v_rel = v1 - v2
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

    # 3. Randomize direction on the unit sphere (Isotropic Scattering)
    # Pick a random point on a sphere surface:
    # cos(theta) is uniform [-1, 1], phi is uniform [0, 2pi]
    cos_theta = 2.0 * rng.random(n_cols) - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * rng.random(n_cols)

    # New direction vector in CoM frame
    nx = sin_theta * np.cos(phi)
    ny = sin_theta * np.sin(phi)
    nz = cos_theta

    # Shape into (N, 3)
    n_vec = np.column_stack((nx, ny, nz))

    # 4. Calculate new velocity for v1
    # v1' = v_cm + (m2 / M_tot) * |v_rel| * n_vec
    return v_cm + (m2 / total_mass) * v_rel_mag * n_vec
