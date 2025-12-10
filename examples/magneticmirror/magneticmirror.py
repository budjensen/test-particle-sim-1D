"""
magneticmirror.py

Demonstration script: ions in a 1D domain with a non-uniform magnetic
field forming a magnetic mirror, and no electric field.

The magnetic field is:
- Strongest at both ends of the domain
- Weakest at the center

Particles initialized near the midplane reflect at appropriate turning
points due to approximate conservation of the adiabatic invariant
(mu * B ≈ constant).

Workflow per timestep
---------------------
1. Compute E(z), B(z) from analytic field functions.
2. Advance particles using the Boris 1D3V integrator.
3. Apply 1D periodic boundary conditions in z (for particles that pass
   through the domain ends in this toy setup).
4. Record diagnostics:
   - global temperature
   - drift velocity
   - density profile n(z)
   - energy distribution f(E)
   - example particle trajectories
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_particle_sim_1d import diagnostics
from test_particle_sim_1d.initialization import constants
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species


def apply_periodic_boundaries(z: np.ndarray, z_min: float, z_max: float) -> None:
    """Apply simple 1D periodic boundary conditions in-place on z positions."""
    length = z_max - z_min
    z[:] = ((z - z_min) % length) + z_min


def make_mirror_field(
    z_min: float,
    z_max: float,
    B_min: float,
    B_max: float,
):
    """
    Build a B(z) function with minimum B at midplane and maximum at both ends.

    We use a simple symmetric parabolic profile:

        B(z) = B_min + (B_max - B_min) * s(z)^2

    where s(z) is a normalized coordinate that is -1 at the left edge,
    +1 at the right edge, and 0 at the midplane.
    """

    z_mid = 0.5 * (z_min + z_max)
    half_length = 0.5 * (z_max - z_min)

    def B_func(z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z)
        # Normalized coordinate: -1 at left, +1 at right, 0 at center
        s = (z_arr - z_mid) / half_length
        Bz = B_min + (B_max - B_min) * s**2

        # Magnetic field along z (can also choose x or y if desired)
        zeros = np.zeros_like(Bz)
        return np.column_stack((zeros, zeros, Bz))

    return B_func


def run_magneticmirror_example() -> None:
    """Run a 1D3V simulation of a magnetic mirror with no electric field."""

    # --- Simulation parameters ---

    # Spatial domain [m]
    z_min = 0.0
    z_max = 0.2

    # Particles
    n_particles = 20000
    species_name = "proton"

    # Time stepping
    dt = 1.0e-9
    n_steps = 4000
    sample_interval = 10  # record diagnostics every N steps

    # Mirror magnetic field parameters
    B_min = 0.05  # Tesla at midplane (weakest)
    B_max = 0.5  # Tesla at both ends (strongest)

    # Diagnostics grid
    n_bins = 50
    z_grid = np.linspace(z_min, z_max, n_bins + 1)
    area = 1.0  # m^2 cross-sectional area

    # Tracer particles for trajectories
    n_tracers = 10

    # --- Initialize ion species ---

    ions = Species(
        q=constants.q_e,
        m=constants.m_p,
        name=species_name,
        capacity=n_particles,
    )

    # Initialize ions near midplane so they bounce between mirror points.
    z_mid = 0.5 * (z_min + z_max)
    # Narrow region around midplane
    z_init_min = z_mid - 0.02 * (z_max - z_min)
    z_init_max = z_mid + 0.02 * (z_max - z_min)

    ions.initialize_maxwellian(
        n=n_particles,
        z_min=z_init_min,
        z_max=z_init_max,
        temperature={"eV": 10.0},  # somewhat energetic
        mean_velocity=0.0,
        seed=42,
    )

    # --- Diagnostics storage ---

    n_samples = n_steps // sample_interval + 1
    time_hist = np.zeros(n_samples)
    T_hist = np.zeros(n_samples)
    drift_hist = np.zeros((n_samples, 3))
    density_hist = np.zeros((n_samples, n_bins))

    tracer_indices = np.arange(min(n_tracers, ions.N))
    traj_hist = np.zeros((n_samples, tracer_indices.size))

    # --- Field functions ---

    def E_func(z: np.ndarray) -> np.ndarray:
        """No electric field (pure magnetic mirror)."""
        z_arr = np.asarray(z)
        zeros = np.zeros_like(z_arr)
        return np.column_stack((zeros, zeros, zeros))

    B_func = make_mirror_field(z_min=z_min, z_max=z_max, B_min=B_min, B_max=B_max)

    # --- Initial diagnostics ---

    sample_idx = 0
    time_hist[sample_idx] = 0.0
    T_hist[sample_idx] = diagnostics.compute_global_temperature(ions)
    drift_hist[sample_idx] = diagnostics.compute_drift_velocity(ions)
    density_hist[sample_idx] = diagnostics.compute_density_profile(ions, z_grid, area)
    traj_hist[sample_idx] = ions.z[tracer_indices]
    sample_idx += 1

    # --- Main time loop ---

    for step in range(1, n_steps + 1):
        # Active particles
        N = ions.N
        z = ions.z[:N].copy()
        v = np.column_stack((ions.vx[:N], ions.vy[:N], ions.vz[:N]))
        q_arr = np.full(N, ions.q)
        m_arr = np.full(N, ions.m)

        # Advance one step with Boris integrator
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

        # Write back
        ions.z[:N] = z
        ions.vx[:N] = v[:, 0]
        ions.vy[:N] = v[:, 1]
        ions.vz[:N] = v[:, 2]

        # Periodic BCs in z
        apply_periodic_boundaries(ions.z[:N], z_min, z_max)

        # Record diagnostics
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

    # --- Final energy distribution (EEDF/IEDF) ---

    energy_centers, energy_pdf = diagnostics.compute_energy_distribution(
        ions, n_bins=100, e_max=None
    )

    # --- Save results ---

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        results_dir / "time_temperature_mirror.csv",
        np.column_stack((time_hist, T_hist)),
        delimiter=",",
        header="time_s,temperature_K",
        comments="",
        fmt="%.8f",
    )

    np.savetxt(
        results_dir / "drift_velocity_mirror.csv",
        np.column_stack((time_hist, drift_hist)),
        delimiter=",",
        header="time_s,vx_mean,vy_mean,vz_mean",
        comments="",
        fmt="%.8f",
    )

    np.savetxt(
        results_dir / "density_profile_mirror.csv",
        density_hist,
        delimiter=",",
        header=",".join([f"bin_{i}" for i in range(density_hist.shape[1])]),
        comments="",
        fmt="%.1f",
    )

    np.savetxt(
        results_dir / "tracer_trajectories_mirror.csv",
        traj_hist,
        delimiter=",",
        header=",".join([f"tracer_{i}" for i in range(traj_hist.shape[1])]),
        comments="",
        fmt="%.4f",
    )

    np.savetxt(
        results_dir / "energy_distribution_mirror.csv",
        np.column_stack((energy_centers, energy_pdf)),
        delimiter=",",
        header="energy_eV,pdf",
        comments="",
        fmt="%.6f",
    )

    np.savez(
        results_dir / "magneticmirror_results.npz",
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
    run_magneticmirror_example()
