"""
diagnostics.py

Tools for calculating physical quantities from particle data in a 1D3V simulation.

This module provides functions to compute:
1. Global quantities (scalar temperature, mean drift).
2. Spatially resolved profiles (density n(z), temperature T(z)).
3. Distribution functions (Energy Distribution Function f(E)).

Functions
---------
compute_global_temperature    : Calculate scalar temperature of a species.
compute_component_temperatures: Calculate temperature for each velocity component.
compute_drift_velocity        : Calculate mean velocity vector.
compute_density_profile       : Calculate number density on a 1D spatial grid.
compute_temperature_profile   : Calculate temperature on a 1D spatial grid.
compute_energy_distribution   : Calculate the Energy Distribution Function (IEDF/EEDF).
"""

from __future__ import annotations

import numpy as np

from . import particles
from .initialization import constants


def compute_global_temperature(species: particles.Species) -> float:
    """
    Calculate the global scalar temperature of a particle species.

    Temperature is derived from the variance of the velocity distribution
    (kinetic energy relative to the mean drift).

    T = (m / 3*kB) * <(v - <v>)^2>

    Parameters
    ----------
    species : Species
        Particle species to analyze.

    Returns
    -------
    float
        Temperature in Kelvin [K]. Returns 0.0 if no particles exist.
    """
    if species.N == 0:
        return 0.0

    # Extract active particle velocities
    vx = species.vx[: species.N]
    vy = species.vy[: species.N]
    vz = species.vz[: species.N]

    # Calculate thermal variances (v - v_drift)^2
    var_x = np.var(vx)  # np.var calculates mean((x - mean(x))**2)
    var_y = np.var(vy)
    var_z = np.var(vz)

    # Total velocity variance (sum of variances for independent dimensions)
    total_variance = var_x + var_y + var_z

    # T = (m * variance) / (k_B * degrees_of_freedom)
    # Using 3 degrees of freedom (3V)
    return (species.m * total_variance) / (3 * constants.kb)


def compute_component_temperatures(
    species: particles.Species,
) -> tuple[float, float, float]:
    """
    Calculate the scalar temperature of a particle species for each velocity component.

    Useful for checking anisotropy (e.g., T_x vs T_y).
    T_i = (m / kB) * <(v_i - <v_i>)^2>  (1 degree of freedom per component)

    Parameters
    ----------
    species : Species
        Particle species to analyze.

    Returns
    -------
    Tuple[float, float, float]
        (Tx, Ty, Tz) in Kelvin [K].
    """
    if species.N == 0:
        return 0.0, 0.0, 0.0

    # Extract active particle velocities
    vx = species.vx[: species.N]
    vy = species.vy[: species.N]
    vz = species.vz[: species.N]

    # Calculate thermal variances
    var_x = np.var(vx)
    var_y = np.var(vy)
    var_z = np.var(vz)

    factor = species.m / constants.kb
    return (factor * var_x, factor * var_y, factor * var_z)


def compute_drift_velocity(species: particles.Species) -> np.ndarray:
    """
    Calculate the global mean drift velocity of a species.

    Parameters
    ----------
    species : Species
        Particle species.

    Returns
    -------
    np.ndarray
        Array [vx_mean, vy_mean, vz_mean] in [m/s].
    """
    if species.N == 0:
        return np.zeros(3)

    return np.array(
        [
            np.mean(species.vx[: species.N]),
            np.mean(species.vy[: species.N]),
            np.mean(species.vz[: species.N]),
        ]
    )


def compute_density_profile(
    species: particles.Species, z_grid: np.ndarray, area: float = 1.0
) -> np.ndarray:
    """
    Calculate the 1D number density profile n(z).

    Parameters
    ----------
    species : Species
        Particle species.
    z_grid : np.ndarray
        Edges of the spatial bins (length M+1).
    area : float, optional
        Cross-sectional area of the simulation domain [m^2]. Default 1.0.

    Returns
    -------
    np.ndarray
        Number density [m^-3] at each bin center (length M).
    """
    if species.N == 0:
        return np.zeros(len(z_grid) - 1)

    z_p = species.z[: species.N]
    weights = species.weight[: species.N]

    # Histogram particles onto the grid
    # "weights" allows for handling super-particles with different weights
    counts, _ = np.histogram(z_p, bins=z_grid, weights=weights)

    # Calculate bin volumes
    dz = np.diff(z_grid)
    volumes = dz * area

    # n = N / Volume
    return counts / volumes


def compute_temperature_profile(
    species: particles.Species, z_grid: np.ndarray
) -> np.ndarray:
    """
    Calculate the 1D temperature profile T(z).

    Parameters
    ----------
    species : Species
        Particle species.
    z_grid : np.ndarray
        Edges of the spatial bins (length M+1).

    Returns
    -------
    np.ndarray
        Temperature [K] at each bin center (length M).
        Returns 0.0 in bins with < 2 particles.
    """
    n_bins = len(z_grid) - 1
    T_profile = np.zeros(n_bins)

    if species.N == 0:
        return T_profile

    z_p = species.z[: species.N]
    vx = species.vx[: species.N]
    vy = species.vy[: species.N]
    vz = species.vz[: species.N]

    # Digitize finds which bin index each particle belongs to
    bin_indices = np.digitize(z_p, z_grid) - 1

    for i in range(n_bins):
        # Mask for particles in the current bin
        mask = bin_indices == i

        # Need at least 2 particles to calculate variance/temperature sensibly
        if np.count_nonzero(mask) < 2:
            T_profile[i] = 0.0
            continue

        vx_bin = vx[mask]
        vy_bin = vy[mask]
        vz_bin = vz[mask]

        # Calculate local variance (thermal energy)
        # Using ddof=1 for sample variance if N is small, but np.var default is population
        # Consistent with global: mean((v - <v>)^2)
        var = np.var(vx_bin) + np.var(vy_bin) + np.var(vz_bin)

        T_profile[i] = (species.m * var) / (3 * constants.kb)

    return T_profile


def compute_energy_distribution(
    species: particles.Species, n_bins: int = 50, e_max: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Energy Distribution Function (EDF) of the species.

    Plots f(E) vs E, where E is kinetic energy.

    Parameters
    ----------
    species : Species
        Particle species.
    n_bins : int, optional
        Number of histogram bins, default 50.
    e_max : float | None, optional
        Maximum energy [eV] to include in histogram. If None, uses max particle energy.

    Returns
    -------
    bin_centers : np.ndarray
        Energy values [eV] (length n_bins).
    pdf : np.ndarray
        Probability Density Function value (normalized such that integral is 1).
    """
    if species.N == 0:
        return np.array([]), np.array([])

    vx = species.vx[: species.N]
    vy = species.vy[: species.N]
    vz = species.vz[: species.N]

    # Calculate kinetic energy in Joules: 0.5 * m * v^2
    v2 = vx**2 + vy**2 + vz**2
    E_joules = 0.5 * species.m * v2

    # Convert to eV
    E_eV = E_joules / constants.q_e

    # Determine range
    if e_max is None:
        e_max = np.max(E_eV)

    # Compute Histogram
    # density=True normalizes it so the integral over the range is 1
    hist, bin_edges = np.histogram(E_eV, bins=n_bins, range=(0, e_max), density=True)

    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, hist
