"""
Monte Carlo Collision (MCC) module for 1D3V particle simulation.

This module implements the null-collision Monte Carlo method for particle-neutral
collisions in a plasma simulation. It handles multiple collision types including
elastic scattering, ionization, charge exchange, and excitation collisions.

The null-collision method evaluates only a subset of particles at each timestep,
selecting candidates based on a maximum collision frequency. This approach is
more efficient than evaluating all particles, especially for low collision rates
or large particle counts.

Key Features
------------
- Null-collision Monte Carlo method for efficient collision processing
- Support for multiple collision types: elastic, ionization, charge exchange, excitation
- Energy-dependent cross-sections via interpolation
- Isotropic scattering in center-of-mass frame
- Proper handling of inelastic energy losses

Classes
-------
MCCollision : Main class for handling Monte Carlo collisions

Notes
-----
The collision probability for each process is determined by:
    P_coll = (n_g * sigma(E) * v_rel) / nu_max

where:
    - n_g is the neutral density
    - sigma(E) is the energy-dependent cross-section
    - v_rel is the relative velocity magnitude
    - nu_max is the maximum collision frequency across all processes

References
----------
.. [1] Birdsall, C. K., & Langdon, A. B. (2004). Plasma Physics via Computer
       Simulation. CRC Press.
.. [2] Vahedi, V., & Surendra, M. (1995). A Monte Carlo collision model for
       the particle-in-cell method: applications to argon and oxygen discharges.
       Computer Physics Communications, 87(1-2), 179-198.

Examples
--------
Set up collision model for electrons in argon gas:

>>> import numpy as np
>>> from test_particle_sim_1d.particles import Species
>>> from test_particle_sim_1d.collisions import MCCollision
>>>
>>> electrons = Species(q=-1.6e-19, m=9.11e-31, name="electron", capacity=1000)
>>> electrons.initialize_uniform(n=100, z_min=-0.01, z_max=0.01, v0=1e6, seed=42)
>>>
>>> # Define cross-section data (energy in eV, cross-section in m^2)
>>> energy = np.linspace(0, 100, 1000)
>>> elastic_xsec = np.full_like(energy, 5e-20)
>>>
>>> collisions = MCCollision(
...     species=electrons,
...     dt=1e-11,
...     neutral_density=1e20,
...     neutral_temp=300,
...     neutral_mass=6.63e-26,  # Argon mass
...     elastic_cross_section=(energy, elastic_xsec)
... )
>>>
>>> # Apply collisions at each timestep
>>> collisions.do_collisions(seed=42)

"""

from __future__ import annotations

import numpy as np

from . import particles
from .initialization import constants


class MCCollision:
    """
    Monte Carlo Collision handler using the null-collision method.

    This class manages all collision processes for a particle species interacting
    with a background neutral gas. It uses the null-collision technique to efficiently
    evaluate collisions by selecting a subset of candidate particles based on the
    maximum collision frequency, then accepting or rejecting collisions based on
    the actual cross-section values.

    The null-collision method is particularly efficient when:
    - The collision frequency is low relative to the inverse timestep
    - The number of particles is large
    - Cross-sections vary significantly with energy

    Attributes
    ----------
    species : particles.Species
        The particle species undergoing collisions.
    dt : float
        Simulation timestep [s].
    n_g : float
        Neutral gas density [m^-3].
    T_g : float
        Neutral gas temperature [K].
    m_g : float
        Mass of neutral gas particles [kg].
    collision_processes : list[bool]
        Flags indicating which collision types are active.
        Index 0: elastic, 1: ionization, 2: charge exchange, 3: excitation.
    energy : list[np.ndarray]
        Energy arrays for each collision process [eV].
    cross_section : list[np.ndarray]
        Cross-section arrays for each collision process [m^2].
    threshold_energy : list[float]
        Energy thresholds for inelastic processes [eV].
    nu_max : float
        Maximum collision frequency across all processes [s^-1].
    max_frac_collisions : float
        Maximum fraction of particles that can collide in one timestep.
    product_species : particles.Species | None
        Species for particles created by ionization (ions from electron impact).

    See Also
    --------
    particles.Species : Particle species container
    particles.sample_maxwellian : Generates Maxwellian velocity distribution

    Notes
    -----
    The collision processes are ordered as:
        0. Elastic scattering
        1. Ionization (electron impact)
        2. Charge exchange
        3. Excitation

    The maximum collision frequency is calculated from:
        nu_max = max(n_g * sqrt(2 * E * q_e / m) * sigma_total(E))

    This ensures that all real collisions are captured by the null-collision
    method, with "null" collisions (rejected candidates) making up the difference.

    """

    def __init__(
        self,
        species: particles.Species,
        dt: float,
        neutral_density: float,
        neutral_temp: float,
        neutral_mass: float,
        elastic_cross_section: tuple[np.ndarray, np.ndarray] | None = None,
        charge_exchange_cross_section: tuple[np.ndarray, np.ndarray] | None = None,
        ionization_cross_section: tuple[np.ndarray, np.ndarray] | None = None,
        excitation_cross_section: tuple[np.ndarray, np.ndarray] | None = None,
        product_species: particles.Species | None = None,
    ):
        """
        Parameters
        ----------
        species : particles.Species
            The particle species undergoing collisions.
        dt : float
            Simulation timestep [s].
        neutral_density : float
            Number density of the background neutral gas [m^-3].
        neutral_temp : float
            Temperature of the background gas [K].
        neutral_mass : float
            Mass of a single background gas particle [kg].
        elastic_cross_section : tuple[np.ndarray, np.ndarray] | None, optional
            Tuple of (energy [eV], cross_section [m^2]) arrays for elastic collisions.
            Default is None (no elastic collisions).
        charge_exchange_cross_section : tuple[np.ndarray, np.ndarray] | None, optional
            Tuple of (energy [eV], cross_section [m^2]) arrays for charge exchange.
            Only valid for ion species. Default is None.
        ionization_cross_section : tuple[np.ndarray, np.ndarray] | None, optional
            Tuple of (energy [eV], cross_section [m^2]) arrays for ionization.
            Only valid for electron species. Requires product_species. Default is None.
        excitation_cross_section : tuple[np.ndarray, np.ndarray] | None, optional
            Tuple of (energy [eV], cross_section [m^2]) arrays for excitation.
            Default is None.
        product_species : particles.Species | None, optional
            Species object for ions created by ionization collisions.
            Required if ionization_cross_section is provided. Default is None.

        Raises
        ------
        ValueError
            If ionization_cross_section is provided without product_species.
        ValueError
            If ionization_cross_section is provided for non-electron species.
        ValueError
            If charge_exchange_cross_section is provided for electron species.

        Examples
        --------
        >>> import numpy as np
        >>> from test_particle_sim_1d.particles import Species
        >>> from test_particle_sim_1d.collisions import MCCollision
        >>>
        >>> electrons = Species(q=-1.0, m=1.0, name="electron", capacity=100)
        >>> electrons.initialize_uniform(n=10, z_min=-0.1, z_max=0.1, v0=1.0, seed=42)
        >>>
        >>> energy = np.linspace(0, 10, 100)
        >>> cross_section = np.full_like(energy, 1.0e-19)
        >>> elec_collisions = MCCollision(
        ...     electrons,
        ...     dt=1e-9,
        ...     neutral_density=1e20,
        ...     neutral_temp_K=300,
        ...     neutral_mass=4.65e-26,
        ...     collision_dict={
        ...         "elastic": [energy, cross_section]
        ...     }
        ... )
        """
        self.species = species
        self.dt = dt

        # Get neutral parameters from collisions dict
        self.n_g = neutral_density
        self.T_g = neutral_temp
        self.m_g = neutral_mass

        # Setup collision containers
        num_collisions = 4
        self.collision_processes = [False] * num_collisions
        # Use numpy arrays as default containers so assigning ndarrays is type-consistent
        self.energy: list[np.ndarray] = [np.array([]) for _ in range(num_collisions)]
        self.cross_section: list[np.ndarray] = [
            np.array([]) for _ in range(num_collisions)
        ]
        self.threshold_energy: list[float] = [0.0] * num_collisions

        if elastic_cross_section is not None:
            coll_idx = 0
            self.collision_processes[coll_idx] = True
            self.energy[coll_idx] = np.asarray(elastic_cross_section[0])
            self.cross_section[coll_idx] = np.asarray(elastic_cross_section[1])

        if ionization_cross_section is not None:
            if product_species is None:
                error_msg = (
                    "product_species must be provided for ionization collisions."
                )
                raise ValueError(error_msg)
            if not np.isclose(self.species.m, constants.m_e, atol=0.0, rtol=1e-4):
                error_msg = (
                    "Electrons must be the base species for ionization collisions."
                )
                raise ValueError(error_msg)
            self.product_species = product_species
            coll_idx = 1
            self.collision_processes[coll_idx] = True
            self.energy[coll_idx] = np.asarray(ionization_cross_section[0])
            self.cross_section[coll_idx] = np.asarray(ionization_cross_section[1])
            self.threshold_energy[coll_idx] = ionization_cross_section[0][0]

        if charge_exchange_cross_section is not None:
            if np.isclose(self.species.m, constants.m_e, atol=0.0, rtol=1e-4):
                error_msg = "Electrons cannot be the base species for charge exchange collisions."
                raise ValueError(error_msg)
            coll_idx = 2
            self.collision_processes[coll_idx] = True
            self.energy[coll_idx] = np.asarray(charge_exchange_cross_section[0])
            self.cross_section[coll_idx] = np.asarray(charge_exchange_cross_section[1])

        if excitation_cross_section is not None:
            coll_idx = 3
            self.collision_processes[coll_idx] = True
            self.energy[coll_idx] = np.asarray(excitation_cross_section[0])
            self.cross_section[coll_idx] = np.asarray(excitation_cross_section[1])
            self.threshold_energy[coll_idx] = excitation_cross_section[0][0]

        # Calculate a total cross section from the sum of active collisions
        nu_max_energy = np.arange(0, 100.1, 0.2)
        total_xsection = np.zeros_like(nu_max_energy)
        for ii, active in enumerate(self.collision_processes):
            if not active:
                continue
            energy = self.energy[ii]
            xsection = self.cross_section[ii]
            total_xsection += np.interp(nu_max_energy, energy, xsection, left=0.0)
        self.nu_max = np.max(
            self.n_g
            * np.sqrt(2 * nu_max_energy * constants.q_e / self.species.m)
            * total_xsection
        )

        # Calculate the maximum fraction of collisions that can occur in a timestep
        # This will be used to limit the number of collision evaluations
        self.max_frac_collisions = 1 - np.exp(-self.nu_max * dt)

    def do_collisions(self, seed: int | None = None):
        """
        Apply Monte Carlo collisions using the null-collision method.

        This method implements the null-collision algorithm by:
        1. Selecting a subset of candidate particles based on max_frac_collisions
        2. Calculating collision energies and relative velocities
        3. Determining collision types using cumulative probability
        4. Applying appropriate collision dynamics for each type

        The null-collision method evaluates only the most likely collision candidates
        rather than all particles, improving computational efficiency.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility. If None, uses random state.
            Default is None.

        Returns
        -------
        None
            Modifies species velocities in-place for particles that collide.

        Notes
        -----
        The collision type is determined by comparing a random number to the
        cumulative collision probability:
            nu_cumulative = sum_{i} (n_g * sigma_i(E) * v_rel / nu_max)

        Collision types are processed in order (elastic, ionization, charge
        exchange, excitation) with the first match winning.

        For inelastic collisions (ionization, excitation), the particle energy
        is reduced by the threshold energy before scattering:
            E_final = E_initial - E_threshold

        Examples
        --------
        >>> collisions = MCCollision(species, dt, n_g, T_g, m_g, ...)
        >>> collisions.do_collisions()

        """
        rng = np.random.default_rng(seed)

        # Access active particle data
        # We only care about the first N particles that are alive
        N = self.species.N
        if np.ceil(N * self.max_frac_collisions) == 0:
            return 0

        # Randomly select indices of active particles to consider for collisions
        num_candidates = int(np.ceil(N * self.max_frac_collisions))
        candidate_indices = rng.choice(N, size=num_candidates, replace=False)
        candidate_indices.sort()

        vx = self.species.vx[candidate_indices]
        vy = self.species.vy[candidate_indices]
        vz = self.species.vz[candidate_indices]

        # Sample candidate neutrals for colliding particles
        v_neutral_all = particles.sample_maxwellian(
            n=num_candidates,
            mass=self.m_g,
            temperature={"K": self.T_g},
            mean_velocity=0.0,
            seed=seed,
        )

        # Calculate vector relative velocity for every particle
        v_rel_x = vx - v_neutral_all[:, 0]
        v_rel_y = vy - v_neutral_all[:, 1]
        v_rel_z = vz - v_neutral_all[:, 2]

        # Compute collisionenergy in eV
        v_coll = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
        E_coll = 0.5 * self.species.m * v_coll**2 / constants.q_e

        # Get random number to determine collision type
        random = rng.random(num_candidates)

        nu = np.zeros(num_candidates)

        for ii, active in enumerate(self.collision_processes):
            if not active:
                continue

            sigma = np.interp(E_coll, self.energy[ii], self.cross_section[ii], left=0.0)

            nu += self.n_g * sigma * v_coll / self.nu_max

            active_collision_mask = random <= nu
            if not np.any(active_collision_mask):
                continue
            active_indices = candidate_indices[active_collision_mask]

            match ii:
                case 0:
                    # Elastic collision
                    v_in = np.column_stack(
                        (
                            vx[active_collision_mask],
                            vy[active_collision_mask],
                            vz[active_collision_mask],
                        )
                    )

                    v_out = self.scatter_isotropic_3d(
                        v1=v_in,
                        v2=v_neutral_all[active_collision_mask],
                        m1=self.species.m,
                        m2=self.m_g,
                        rng=rng,
                    )

                    self.species.vx[active_indices] = v_out[:, 0]
                    self.species.vy[active_indices] = v_out[:, 1]
                    self.species.vz[active_indices] = v_out[:, 2]
                case 1:
                    # Ionization collision
                    # Account for inelastic energy loss
                    scale_fac = self.get_energy_loss_scale(
                        self.threshold_energy[ii],
                        E_coll[active_collision_mask],
                        v_coll[active_collision_mask],
                    )
                    # Scale velocity down by sqrt(2) so total energy is conserved
                    # when this electron shares energy with the ionized electron
                    vy[active_collision_mask] *= scale_fac / np.sqrt(2)
                    vz[active_collision_mask] *= scale_fac / np.sqrt(2)
                    vx[active_collision_mask] *= scale_fac / np.sqrt(2)

                    # Create product particles with electron positions
                    z_product = self.species.z[active_indices]
                    v_ion = v_neutral_all[active_collision_mask]
                    # Both electrons get half of the remaining energy
                    v_electron = np.column_stack(
                        (
                            vx[active_collision_mask],
                            vy[active_collision_mask],
                            vz[active_collision_mask],
                        )
                    )
                    # Adding to the product species increases species.N by num_created
                    # New particles live in the range [species.N - num_created : species.N]
                    num_created = len(z_product)
                    self.species.add_particles(z_product, v_electron)
                    self.product_species.add_particles(z_product, v_ion)

                    v_in = np.column_stack(
                        (
                            vx[active_collision_mask],
                            vy[active_collision_mask],
                            vz[active_collision_mask],
                        )
                    )

                    v_out_incident = self.scatter_isotropic_3d(
                        v1=v_in,
                        v2=v_neutral_all[active_collision_mask],
                        m1=self.species.m,
                        m2=self.m_g,
                        rng=rng,
                    )
                    self.species.vx[active_indices] = v_out_incident[:, 0]
                    self.species.vy[active_indices] = v_out_incident[:, 1]
                    self.species.vz[active_indices] = v_out_incident[:, 2]

                    v_out_product = self.scatter_isotropic_3d(
                        v1=v_in,
                        v2=v_neutral_all[active_collision_mask],
                        m1=self.species.m,
                        m2=self.m_g,
                        rng=rng,
                    )
                    new_electron_slice = slice(
                        self.species.N - num_created, self.species.N
                    )
                    self.species.vx[new_electron_slice] = v_out_product[:, 0]
                    self.species.vy[new_electron_slice] = v_out_product[:, 1]
                    self.species.vz[new_electron_slice] = v_out_product[:, 2]
                case 2:
                    # Charge exchange collision
                    # Swap ion and neutral velocities
                    self.species.vx[active_indices] = v_neutral_all[
                        active_collision_mask, 0
                    ]
                    self.species.vy[active_indices] = v_neutral_all[
                        active_collision_mask, 1
                    ]
                    self.species.vz[active_indices] = v_neutral_all[
                        active_collision_mask, 2
                    ]
                case 3:
                    # Excitation collision
                    # Account for inelastic energy loss
                    scale_fac = self.get_energy_loss_scale(
                        self.threshold_energy[ii],
                        E_coll[active_collision_mask],
                        v_coll[active_collision_mask],
                    )
                    vx[active_collision_mask] *= scale_fac
                    vy[active_collision_mask] *= scale_fac
                    vz[active_collision_mask] *= scale_fac

                    v_in = np.column_stack(
                        (
                            vx[active_collision_mask],
                            vy[active_collision_mask],
                            vz[active_collision_mask],
                        )
                    )

                    v_out = self.scatter_isotropic_3d(
                        v1=v_in,
                        v2=v_neutral_all[active_collision_mask],
                        m1=self.species.m,
                        m2=self.m_g,
                        rng=rng,
                    )

                    self.species.vx[active_indices] = v_out[:, 0]
                    self.species.vy[active_indices] = v_out[:, 1]
                    self.species.vz[active_indices] = v_out[:, 2]

            # After each collision type, set random number to 1.0
            random[active_collision_mask] = 1.0

        return None

    def get_energy_loss_scale(
        self,
        E_th: float,
        E_coll: np.ndarray,
        v_coll: np.ndarray,
    ):
        """
        Calculate velocity scaling factor for inelastic collisions.

        For inelastic collisions (ionization, excitation), particles lose energy
        equal to the threshold energy of the process. This method calculates the
        factor by which to scale the particle velocity to account for this loss.

        Parameters
        ----------
        E_th : float
            Threshold energy for the inelastic process [eV].
        E_coll : np.ndarray
            Collision energies of particles before collision [eV].
        v_coll : np.ndarray
            Collision velocities (relative speed magnitudes) [m/s].

        Returns
        -------
        np.ndarray
            Velocity scaling factors (dimensionless). Multiply particle velocity
            components by this factor to account for energy loss.

        Notes
        -----
        The scaling is derived from energy conservation:
            0.5 * m * v_final^2 = E_initial - E_threshold

        Therefore:
            scale = v_final / v_initial = sqrt((E_initial - E_threshold) / E_initial)

        The energy is converted from eV to Joules using the elementary charge.

        Examples
        --------
        >>> E_threshold = 15.76  # Argon ionization threshold [eV]
        >>> E_collision = np.array([20.0, 30.0, 50.0])  # Collision energies [eV]
        >>> v_collision = np.array([2e6, 2.5e6, 3e6])  # Velocities [m/s]
        >>> scale = collisions.get_energy_loss_scale(E_threshold, E_collision, v_collision)
        >>> v_new = v_old * scale  # Apply scaling to velocity

        """
        # Subtract energy penalty (result in eV)
        E_coll_eV = E_coll - E_th

        # Return velocity scale
        return np.sqrt(2.0 * E_coll_eV * constants.q_e / self.species.m) / v_coll

    def scatter_isotropic_3d(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        m1: float | None,
        m2: float | None,
        rng: np.random.Generator,
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
        m1 : float, optional
            Mass of scattering particles. If none is provided,
            defaults to the mass of species 1.
        m2 : float, optional
            Mass of target particles. If none is provided,
            defaults to the neutral mass.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        np.ndarray
            New velocities for species 1, shape (N, 3).
        """
        if m1 is None:
            m1 = self.species.m
        if m2 is None:
            m2 = self.m_g
        n_cols = len(v1)
        total_mass = m1 + m2

        # 1. Calculate Center of Mass Velocity
        v_cm = (m1 * v1 + m2 * v2) / total_mass

        # 2. Calculate Relative Velocity (v_rel = v1 - v2)
        v_rel = v1 - v2
        v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)

        # 3. Isotropically scatter on unit sphere
        # To pick a random point on a sphere surface:
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
