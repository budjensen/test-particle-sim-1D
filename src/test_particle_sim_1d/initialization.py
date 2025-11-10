"""
initialization.py

Defines functions and constants for initializing particle species
and validating simulation parameters.

Functions
---------
validate_gyromotion_timestep : Ensure timestep resolves gyromotion.
validate_thermal_velocity_grid : Check that the gridsize and timestep are valid for a species' thermal velocity.
get_max_collision_frequency : Calculate maximum collision frequency.
validate_collision_frequency_timestep : Ensure timestep is small enough for MCC collision frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from test_particle_sim_1d.particles import Species

import numpy as np


def validate_gyromotion_timestep(dt: float, species: Species, B_max: float):
    """
    Check that the timestep resolves gyromotion.

    Timestep should be small (`T_cyclotron < 200 * dt`) compared to the
    cyclotron period. This function raises a ValueError if the timestep is too large.

    Parameters
    ----------
    dt : float
        Timestep [s]
    species : Species
        Particle species
    B_max : float
        Maximum magnetic field [T]

    Examples
    --------
    >>> from test_particle_sim_1d.initialization import validate_gyromotion_timestep
    >>> from test_particle_sim_1d.particles import Species
    >>> sp = Species(q=1.0, m=1.0)
    >>> validate_gyromotion_timestep(dt=0.01, species=sp, B_max=1.0)
    >>> validate_gyromotion_timestep(dt=1.0, species=sp, B_max=4.0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
        import platform
      File "/Users/brian/semester_files/test-particle-sim-1D/src/test_particle_sim_1d/initialization.py", line 63, in validate_gyromotion_timestep
        raise ValueError(error_msg)
    ValueError: Timestep dt=1.0 is too large to resolve gyromotion (T_cyclotron=1.5707963267948966).

    Raises
    ------
    ValueError
        If dt is not positive or too large to resolve gyromotion
    """
    if dt <= 0:
        error_msg = "Timestep dt must be positive."
        raise ValueError(error_msg)

    # Calculate cyclotron period
    q = species.q
    m = species.m
    omega_c = q * B_max / m
    T_cyclotron = 2 * np.pi / omega_c

    if dt > 0.005 * np.min(T_cyclotron):
        error_msg = f"Timestep dt={dt} is too large to resolve gyromotion (T_cyclotron={T_cyclotron})."
        raise ValueError(error_msg)


def validate_thermal_velocity_grid(
    dt: float, dz: float, temperature: dict[str, float], species: Species
):
    """
    Check that the grid spacing and timestep are valid for a species' thermal velocity.

    Intended for use when the electric field is defined on a grid.

    Parameters
    ----------
    dt : float
        Timestep [s]
    dz : float
        Grid spacing [m]
    temperature : dict[str, float]
        Temperature in Kelvin or electronvolts
    species : Species
        Particle species

    Notes
    -----
    Thermal velocity should be small enough that particles do not traverse more than
    ~40% of a grid cell in one timestep. Raises a ValueError if condition is not met.


    Raises
    ------
    ValueError
        If either dz or dt are not positive, or if the thermal velocity condition is not met
    ValueError
        If temperature dictionary does not contain 'K' or 'eV' key
    """
    if dz <= 0:
        error_msg = "Grid spacing dz must be a positive number."
        raise ValueError(error_msg)
    if dt <= 0:
        error_msg = "Timestep dt must be a positive number."
        raise ValueError(error_msg)

    # Calculate thermal velocity
    if "eV" in temperature:
        v_th = np.sqrt(constants.q_e * temperature["eV"] / species.m)
    elif "K" in temperature:
        v_th = np.sqrt(constants.kb * temperature["K"] / species.m)
    else:
        error_msg = "Temperature dictionary must contain either 'K' or 'eV' key."
        raise ValueError(error_msg)

    if v_th > 0.4 * dz / dt:
        error_msg = f"Grid spacing dz={dz} is too small for thermal velocity v_th={v_th} and timestep dt={dt}."
        raise ValueError(error_msg)


def get_max_collision_frequency(
    energy: np.ndarray,
    cross_section: np.ndarray,
    gas_density: np.ndarray,
    species: Species,
) -> float:
    """
    Calculate the maximum collision frequency for given energy, cross section, and gas density.

    Parameters
    ----------
    energy : np.ndarray
        Cross section energies [eV]
    cross_section : np.ndarray
        Collision cross sections [m^2] versus energy
    gas_density : np.ndarray
        Gas number density [1/m^3] versus position across the simulation
    species : Species
        Particle species

    Returns
    -------
    float
        Maximum collision frequency [Hz]
    """
    n_g = np.max(gas_density)
    return np.max(cross_section * energy * np.sqrt(2 * energy / species.m)) * n_g


def validate_collision_frequency_timestep(dt: float, nu: float):
    """
    Check that timestep is small enough for the given MCC collision frequency.

    Parameters
    ----------
    nu : float
        Collision frequency [Hz]
    dt : float
        Timestep [s]

    Notes
    -----
    dt needs to be small enough that linear expansion of collision
    probability is sufficiently accurate, otherwise MCC algorithm
    will be affected by small changes in timestep

    Raises
    ------
    ValueError
        If nu or dt are not positive, or if the collision frequency condition is not met
    """
    if nu < 0:
        error_msg = "Collision frequency nu must be non-negative."
        raise ValueError(error_msg)
    if dt <= 0:
        error_msg = "Timestep dt must be a positive number."
        raise ValueError(error_msg)

    coll_n = nu * dt
    total_collision_prob = 1.0 - np.exp(-coll_n)

    if coll_n > 0.1:
        error_msg = f"Timestep dt={dt} is too large for collision frequency nu={nu}. Total collision probability={total_collision_prob:.3f} exceeds 0.1."
        raise ValueError(error_msg)


class constants:
    """
    Physical constants for use in the simulation

    Attributes
    ----------
    q_e : float
        Elementary charge [C]
    m_e : float
        Electron mass [kg]
    m_p : float
        Proton mass [kg]
    ep0 : float
        Vacuum permittivity [F/m]
    mu0 : float
        Vacuum permeability [N/A^2]
    c : float
        Speed of light in vacuum [m/s]
    kb : float
        Boltzmann constant [J/K]
    """

    q_e = 1.602176634e-19  # C
    m_e = 9.1093837015e-31  # kg
    m_p = 1.67262192369e-27  # kg
    ep0 = 8.8541878128e-12  # F/m
    mu0 = 1.25663706212e-6  # N/A^2
    c = 299792458.0  # m/s
    kb = 1.380649e-23  # J/K
