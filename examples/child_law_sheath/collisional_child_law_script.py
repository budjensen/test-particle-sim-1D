"""
child_law_sheath/collisional_child_law_script.py

Demonstration script: 300K ions traversing a collisional sheath.
Output data shows the energy distribution of ions arriving at the wall.

Workflow per timestep
---------------------
1. Compute E(z) from analytic field functions.
2. Advance particles using the Boris 1D3V integrator.
3. Apply 1D absorbing boundary conditions in z.
4. Apply Monte Carlo collisions with neutrals.
5. Record diagnostics:
   - ion energy distribution f(E) at the boundary
   - example particle trajectories
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_particle_sim_1d import collisions, fields
from test_particle_sim_1d.initialization import (
    constants,
    validate_collision_frequency_timestep,
    validate_thermal_velocity_grid,
)
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species, sample_maxwellian

argon_cross_section_data = np.load("cross_sections/argon_momentum_transfer.npy")
torr_to_m3 = 3.223e22  # m^-3 per Torr at 300 K


def store_and_remove_particles(
    species: Species, z_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Remove particles and return a buffer of particles that have exited the domain at z_max.

    Parameters
    ----------
    species : Species
        Particle species containing positions and velocities.
    z_max : float
        Maximum z position for the simulation domain.

    Returns
    -------
    buffer : np.ndarray
        Buffer containing the velocities of particles that have exited the domain.
        Shape is (N, 3) where N is the number of particles removed.
    remove_idx : np.ndarray
        Indices of the particles that were removed from the species.
    """
    # Create buffer
    remove_mask = species.z >= z_max
    buffer = np.column_stack(
        (species.vx[remove_mask], species.vy[remove_mask], species.vz[remove_mask])
    )

    # Remove particles
    remove_idx = np.where(remove_mask)[0]
    species.remove_particles(remove_idx)

    return buffer, remove_idx


def run_child_sheath_example() -> None:
    """Run a 1D3V simulation with a sheath E-field and elastic collisions."""

    # --- Simulation parameters ---

    # Spatial domain [m]
    z_min = 0.0
    z_max = 10e-3

    # Particles
    n_particles = 2000
    species_name = "Argon"

    # Time stepping
    dt = 5.0e-10
    n_steps = 20000
    sample_interval = 100  # record diagnostics every N steps

    # Field: child law sheath Ez
    V0 = 100  # V

    # Background neutral gas for collisions
    neutral_density = 0.1 * torr_to_m3  # m^-3
    neutral_temp_K = 300.0  # K
    neutral_mass = 39.948 * constants.m_p  # Argon ions

    # Density profile / diagnostics grid
    n_bins = 100
    z_grid = np.linspace(z_min, z_max, n_bins + 1)

    # Choose a small subset of tracer particles for trajectories
    n_tracers = 10

    # --- Initialize ions ---
    ions = Species(
        q=constants.q_e,
        m=39.948 * constants.m_p,
        name=species_name,
        capacity=n_particles,
    )
    # Initialize ion velocities at 300 K and at the lower (plasma) boundary
    v_init = sample_maxwellian(
        n=n_particles,
        mass=ions.m,
        temperature={"K": 300.0},
        mean_velocity=0.0,
        seed=42,
    )
    # Force ions to all begin with +z directed initial velocity (towards the sheath)
    v_init[:, 2] = np.abs(v_init[:, 2])
    z_init = np.zeros(n_particles)
    ions.add_particles(z_init, v_init)

    # --- Set up collision handler (elastic ion-neutral collisions) ---

    # Simple constant elastic cross section as a function of energy
    # Energy grid in eV
    energy_grid = argon_cross_section_data[0]
    sigma_elastic = argon_cross_section_data[1]

    collision_handler = collisions.MCCollision(
        species=ions,
        dt=dt,
        neutral_density=neutral_density,
        neutral_temp=neutral_temp_K,
        neutral_mass=neutral_mass,
        elastic_cross_section=(energy_grid, sigma_elastic),
    )

    # --- Verify timestep and cell size are appropriate ---

    validate_thermal_velocity_grid(
        dz=(z_max - z_min) / n_bins,
        dt=dt,
        temperature={"K": 300.0},
        species=ions,
    )

    validate_collision_frequency_timestep(
        dt=dt,
        nu=collision_handler.nu_max,
    )

    # --- Diagnostics storage ---

    n_samples = n_steps // sample_interval + 1
    time_hist = np.zeros(n_samples)
    lost_particles = np.zeros(shape=(n_particles, 3))
    lost_particle_count = 0

    # Example trajectories: z positions of a few tracer particles
    tracer_indices = np.arange(min(n_tracers, ions.N))
    traj_hist = np.zeros((n_samples, tracer_indices.size))

    # --- Define field functions ---

    def E_child_law(z: np.ndarray, V0: float, zmin: float, zmax: float) -> np.ndarray:
        """
        Simple child law sheath electric field that increases with z^(1/3) near the wall.
        """
        s = zmax - zmin
        with np.errstate(invalid="ignore"):
            Ez = (4 / 3) * V0 / s * (z / s) ** (1 / 3)
        return np.column_stack((np.zeros_like(Ez), np.zeros_like(Ez), Ez))

    def B_zero(z: np.ndarray) -> np.ndarray:
        """
        Zero magnetic field.
        """
        return fields.B_uniform(z, B0=0, direction="y")

    # --- Main time integration loop ---

    sample_idx = 0

    # Record initial diagnostics
    time_hist[sample_idx] = 0.0
    traj_hist[sample_idx] = ions.z[tracer_indices]
    sample_idx += 1

    for step in range(1, n_steps + 1):
        # Extract active particle arrays
        N = ions.N
        z = ions.z[:N].copy()
        v = np.column_stack((ions.vx[:N], ions.vy[:N], ions.vz[:N]))
        q_arr = np.full(N, ions.q)
        m_arr = np.full(N, ions.m)

        # Advance particles one step with Boris integrator
        z, v = boris_1d3v_z(
            z,
            v,
            q_arr,
            m_arr,
            lambda z: E_child_law(z, V0=V0, zmin=z_min, zmax=z_max),
            B_zero,
            dt,
            n_steps=1,
            record_history=False,
        )

        # Write back updated positions and velocities
        ions.z[:N] = z
        ions.vx[:N] = v[:, 0]
        ions.vy[:N] = v[:, 1]
        ions.vz[:N] = v[:, 2]

        # Remove particles that have exited the domain
        lost_particles_buffer, removed_indices = store_and_remove_particles(ions, z_max)
        if len(lost_particles_buffer) > 0:
            n_to_store = len(lost_particles_buffer)
            lost_particles[
                lost_particle_count : lost_particle_count + n_to_store, :
            ] = lost_particles_buffer
            lost_particle_count += n_to_store
            if removed_indices.size > 0:
                # Replace the tracer indices with self.N if they were removed
                # and make sure self.N is set to z = z_max for plotting purposes
                for i, tracer_idx in enumerate(tracer_indices):
                    if tracer_idx in removed_indices:
                        tracer_indices[i] = ions.N
                ions.z[ions.N] = z_max

        # Apply Monte Carlo collisions
        collision_handler.do_collisions(seed=None)

        # Record diagnostics at the desired interval
        if step % sample_interval == 0:
            t = step * dt
            time_hist[sample_idx] = t
            traj_hist[sample_idx] = ions.z[tracer_indices]
            sample_idx += 1

        if lost_particle_count == n_particles:
            # All particles have exited the domain; end simulation early
            break

    # --- Final energy distribution (EEDF/IEDF) ---

    energy_bins = np.linspace(0, V0 + 5, 100)  # eV
    lost_energies = (
        0.5
        * ions.m
        * (
            lost_particles[:lost_particle_count, 0] ** 2
            + lost_particles[:lost_particle_count, 1] ** 2
            + lost_particles[:lost_particle_count, 2] ** 2
        )
        / constants.q_e
    )
    wall_edf = np.histogram(
        lost_energies,
        bins=energy_bins,
        density=True,
    )
    energy_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    # --- Save results ---

    # Remove any zero entries at the end of the trajectories array
    traj_hist = traj_hist[:sample_idx, :]
    time_hist = time_hist[:sample_idx]

    results_dir = Path(__file__).resolve().parent / "collisional_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSVs (optional, mostly for inspection)
    np.savetxt(
        results_dir / "tracer_trajectories.csv",
        traj_hist,
        delimiter=",",
        header=",".join([f"tracer_{i}" for i in range(traj_hist.shape[1])]),
        comments="",
        fmt="%.4f",
    )

    np.savetxt(
        results_dir / "energy_distribution.csv",
        np.column_stack((energy_centers, wall_edf[0])),
        delimiter=",",
        header="energy_eV,pdf",
        comments="",
        fmt="%.6f",
    )

    # NPZ bundle specific to this E+B example
    np.savez(
        results_dir / "child_sheath_results.npz",
        time=time_hist,
        z_grid=z_grid,
        tracer_indices=tracer_indices,
        tracer_trajectories=traj_hist,
        energy_centers=energy_centers,
        energy_pdf=wall_edf[0],
    )


if __name__ == "__main__":
    run_child_sheath_example()
