"""
uniformE.py

Demonstration script: ions in a 1D domain with a uniform electric field
and Monte Carlo elastic collisions with a background neutral gas.

Workflow per timestep
---------------------
1. Compute E(z), B(z) from analytic field functions.
2. Advance particles using the Boris 1D3V integrator.
3. Apply 1D periodic boundary conditions in z.
4. Apply Monte Carlo collisions with neutrals (elastic only).
5. Record diagnostics:
   - global temperature
   - drift velocity
   - density profile n(z)
   - energy distribution f(E)
   - example particle trajectories
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_particle_sim_1d import collisions, diagnostics, fields
from test_particle_sim_1d.initialization import constants
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species


def apply_periodic_boundaries(z: np.ndarray, z_min: float, z_max: float) -> None:
    """Apply simple 1D periodic boundary conditions in-place on z positions."""
    L = z_max - z_min
    # Wrap positions back into [z_min, z_max)
    z[:] = ((z - z_min) % L) + z_min


def run_uniformE_example() -> None:
    """Run a simple 1D3V simulation with uniform E-field and elastic collisions."""

    # Simulation parameters

    # Spatial domain [m]
    z_min = 0.0
    z_max = 0.1

    # Particles
    n_particles = 20000
    species_name = "proton"

    # Time stepping
    dt = 1.0e-9
    n_steps = 2000
    sample_interval = 10  # record diagnostics every N steps

    # Fields: uniform Ez, no magnetic field
    E0 = 5.0e3  # V/m, accelerates ions along +z
    B0 = 0.0  # Tesla

    # Background neutral gas for collisions
    neutral_density = 2.0e21  # m^-3
    neutral_temp_K = 300.0  # K
    neutral_mass = constants.m_p  # assume same mass as ions (e.g. hydrogen)

    # Density profile / diagnostics grid
    n_bins = 50
    z_grid = np.linspace(z_min, z_max, n_bins + 1)
    area = 1.0  # m^2 cross-sectional area

    # Choose a small subset of tracer particles for trajectories
    n_tracers = 10

    # Initialize ion species

    ions = Species(
        q=constants.q_e,
        m=constants.m_p,
        name=species_name,
        capacity=n_particles,
    )
    # Initialize ions with Maxwellian velocities at ~1 eV and uniform z distribution
    ions.initialize_maxwellian(
        n=n_particles,
        z_min=z_min,
        z_max=z_max,
        temperature={"eV": 1.0},
        mean_velocity=0.0,
        seed=42,
    )

    # Set up collision handler (elastic ion-neutral collisions)

    # Simple constant elastic cross section as a function of energy
    # Energy grid in eV
    energy_grid = np.array([0.0, 10.0])
    sigma_elastic = np.array([1.0e-19, 1.0e-19])

    collision_handler = collisions.MCCollision(
        species=ions,
        dt=dt,
        neutral_density=neutral_density,
        neutral_temp=neutral_temp_K,
        neutral_mass=neutral_mass,
        elastic_cross_section=(energy_grid, sigma_elastic),
    )

    # Diagnostics storage

    n_samples = n_steps // sample_interval + 1
    time_hist = np.zeros(n_samples)
    T_hist = np.zeros(n_samples)
    drift_hist = np.zeros((n_samples, 3))
    density_hist = np.zeros((n_samples, n_bins))

    # Example trajectories: z positions of a few tracer particles
    tracer_indices = np.arange(min(n_tracers, ions.N))
    traj_hist = np.zeros((n_samples, tracer_indices.size))

    # Define field functions (compatible with boris_1d3v_z)

    def E_func(z: np.ndarray) -> np.ndarray:
        # Uniform Ez field along +z
        return fields.E_uniform(z, E0=E0, direction="z")

    def B_func(z: np.ndarray) -> np.ndarray:
        # No magnetic field in this example
        return fields.B_uniform(z, B0=B0, direction="z")

    # 6. Main time integration loop

    sample_idx = 0

    # Record initial diagnostics
    time_hist[sample_idx] = 0.0
    T_hist[sample_idx] = diagnostics.compute_global_temperature(ions)
    drift_hist[sample_idx] = diagnostics.compute_drift_velocity(ions)
    density_hist[sample_idx] = diagnostics.compute_density_profile(ions, z_grid, area)
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
            E_func,
            B_func,
            dt,
            n_steps=1,
            record_history=False,
        )

        # Write back updated positions and velocities
        ions.z[:N] = z
        ions.vx[:N] = v[:, 0]
        ions.vy[:N] = v[:, 1]
        ions.vz[:N] = v[:, 2]

        # Apply periodic boundary conditions in z
        apply_periodic_boundaries(ions.z[:N], z_min, z_max)

        # Apply Monte Carlo collisions (elastic only in this example)
        collision_handler.do_collisions(seed=None)

        # Record diagnostics at the desired interval
        if step % sample_interval == 0:
            t = step * dt
            time_hist[sample_idx] = t
            T_hist[sample_idx] = diagnostics.compute_global_temperature(ions)
            drift_hist[sample_idx] = diagnostics.compute_drift_velocity(ions)
            density_hist[sample_idx] = diagnostics.compute_density_profile(
                ions, z_grid, area
            )
            traj_hist[sample_idx] = ions.z[tracer_indices]
            sample_idx += 1

    # Final energy distribution (EEDF/IEDF)

    energy_centers, energy_pdf = diagnostics.compute_energy_distribution(
        ions, n_bins=100, e_max=None
    )

    # Save results

    # Folder to save results into
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save time + temperature
    np.savetxt(
        results_dir / "time_temperature.csv",
        np.column_stack((time_hist, T_hist)),
        delimiter=",",
        header="time_s,temperature_K",
        comments="",
        fmt="%.8f",
    )

    # Save drift velocity
    np.savetxt(
        results_dir / "drift_velocity.csv",
        np.column_stack((time_hist, drift_hist)),
        delimiter=",",
        header="time_s,vx_mean,vy_mean,vz_mean",
        comments="",
        fmt="%.8f",
    )

    # Save density profile (each column is a spatial bin)
    np.savetxt(
        results_dir / "density_profile.csv",
        density_hist,
        delimiter=",",
        header=",".join([f"bin_{i}" for i in range(density_hist.shape[1])]),
        comments="",
        fmt="%.1f",
    )

    # Save tracer trajectories
    np.savetxt(
        results_dir / "tracer_trajectories.csv",
        traj_hist,
        delimiter=",",
        header=",".join([f"tracer_{i}" for i in range(traj_hist.shape[1])]),
        comments="",
        fmt="%.4f",
    )

    # Save final energy distribution (EEDF/IEDF)
    np.savetxt(
        results_dir / "energy_distribution.csv",
        np.column_stack((energy_centers, energy_pdf)),
        delimiter=",",
        header="energy_eV,pdf",
        comments="",
        fmt="%.6f",
    )

    np.savez(
        results_dir / "uniformE_results.npz",
        time=time_hist,
        temperature=T_hist,
        drift_velocity=drift_hist,
        z_grid=z_grid,
        density_profile=density_hist,
        tracer_indices=tracer_indices,
        tracer_trajectories=traj_hist,
        energy_centers=energy_centers,
        energy_pdf=energy_pdf,
    )


if __name__ == "__main__":
    run_uniformE_example()
