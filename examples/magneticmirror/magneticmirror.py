"""
magneticmirror.py

Textbook magnetic mirror example in a 1D domain.

Model
-----
Ions move along a 1D coordinate z in a non-uniform magnetic field B(z).
The field:

- is weakest at the midplane z = (z_min + z_max)/2,
- is strongest near both ends of the domain.

The ions execute gyromotion in the transverse plane and experience
a parallel "magnetic mirror" force due to approximate conservation of
the first adiabatic invariant,

    mu = m v_perp^2 / (2 B)   (approximately constant).

The effective parallel equation of motion is

    dv_parallel/dt = -(mu / m) dB/dz.

In this script we:
1. Use the Boris 1D3V integrator to advance the full 3-velocity v.
2. Add the mirror-force kick to v_z after each Boris step.
3. Apply 1D periodic boundary conditions in z, representing a
   repeating magnetic mirror cell.
4. Record diagnostics:
   - global temperature
   - drift velocity
   - density profile n(z)
   - energy distribution f(E)
   - tracer particle trajectories

The resulting trajectories show a mix of:
- trapped (bouncing) particles, which reflect before reaching
  the strongest-B regions,
- passing particles, which stream through the domain and re-enter
  due to periodic boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_particle_sim_1d import diagnostics
from test_particle_sim_1d.initialization import constants
from test_particle_sim_1d.integrators import boris_1d3v_z
from test_particle_sim_1d.particles import Species

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def apply_periodic_boundaries(z: np.ndarray, z_min: float, z_max: float) -> None:
    """Apply simple 1D periodic boundary conditions in-place on z positions."""
    length = z_max - z_min
    z[:] = ((z - z_min) % length) + z_min


# ---------------------------------------------------------------------------
# Magnetic mirror field: B(z), |B(z)|, and dB/dz
# ---------------------------------------------------------------------------


def make_mirror_field(
    z_min: float,
    z_max: float,
    B_min: float,
    B_max: float,
):
    """
    Build a magnetic mirror with minimum field at the midplane and
    maximum at both ends.

    We use a simple symmetric parabolic profile for the magnitude:

        B(z) = B_min + (B_max - B_min) * s(z)^2,

    where s(z) is a normalized coordinate that is -1 at the left edge,
    +1 at the right edge, and 0 at the midplane.
    """

    z_mid = 0.5 * (z_min + z_max)
    half_length = 0.5 * (z_max - z_min)
    delta_B = B_max - B_min

    def B_vec(z: np.ndarray) -> np.ndarray:
        """Full magnetic-field vector B(z) for the Boris integrator."""
        z_arr = np.asarray(z)
        s = (z_arr - z_mid) / half_length
        Bz = B_min + delta_B * s**2
        zeros = np.zeros_like(Bz)
        return np.column_stack((zeros, zeros, Bz))

    def B_mag(z: np.ndarray) -> np.ndarray:
        """Magnetic-field magnitude B(z)."""
        z_arr = np.asarray(z)
        s = (z_arr - z_mid) / half_length
        return B_min + delta_B * s**2

    def dB_dz(z: np.ndarray) -> np.ndarray:
        """Spatial derivative dB/dz along the mirror axis."""
        z_arr = np.asarray(z)
        s = (z_arr - z_mid) / half_length
        return 2.0 * delta_B * s / half_length

    return B_vec, B_mag, dB_dz


# ---------------------------------------------------------------------------
# Main example driver
# ---------------------------------------------------------------------------


def run_magneticmirror_example() -> None:
    """Run a 1D3V simulation of ions in a magnetic mirror with no E-field."""

    # --- Simulation parameters ------------------------------------------------

    # Spatial domain [m]
    z_min = 0.0
    z_max = 0.2

    # Particles
    n_particles = 20_000
    species_name = "proton"

    # Time stepping
    dt = 1.0e-9
    n_steps = 5_000  # much lighter run, still enough for bounces
    sample_interval = 10  # record every 10 steps

    # Mirror magnetic field parameters (strong mirror ratio)
    B_min = 0.03  # Tesla at midplane (weakest)
    B_max = 1.0  # Tesla at both ends (strongest)

    # Diagnostics grid
    n_bins = 50
    z_grid = np.linspace(z_min, z_max, n_bins + 1)
    area = 1.0  # m^2 cross-sectional area

    # Tracer particles for trajectories
    n_tracers = 10

    # --- Initialize ion species ----------------------------------------------

    ions = Species(
        q=constants.q_e,
        m=constants.m_p,
        name=species_name,
        capacity=n_particles,
    )

    # Initialize ions near the midplane.
    z_mid = 0.5 * (z_min + z_max)
    z_init_min = z_mid - 0.02 * (z_max - z_min)
    z_init_max = z_mid + 0.02 * (z_max - z_min)

    ions.initialize_maxwellian(
        n=n_particles,
        z_min=z_init_min,
        z_max=z_init_max,
        temperature={"eV": 15.0},  # a bit hotter than before
        mean_velocity=0.0,
        seed=42,
    )

    # --- Diagnostics storage --------------------------------------------------

    n_samples = n_steps // sample_interval + 1
    time_hist = np.zeros(n_samples)
    T_hist = np.zeros(n_samples)
    drift_hist = np.zeros((n_samples, 3))
    density_hist = np.zeros((n_samples, n_bins))

    tracer_indices = np.arange(min(n_tracers, ions.N))
    traj_hist = np.zeros((n_samples, tracer_indices.size))

    # --- Field functions ------------------------------------------------------

    def E_func(z: np.ndarray) -> np.ndarray:
        """No electric field (pure magnetic mirror)."""
        z_arr = np.asarray(z)
        zeros = np.zeros_like(z_arr)
        return np.column_stack((zeros, zeros, zeros))

    B_func, B_mag_func, dB_dz_func = make_mirror_field(
        z_min=z_min,
        z_max=z_max,
        B_min=B_min,
        B_max=B_max,
    )

    # Strength factor to make the mirror effect visually clear
    mirror_strength = 5.0

    # --- Initial diagnostics --------------------------------------------------

    sample_idx = 0
    time_hist[sample_idx] = 0.0
    T_hist[sample_idx] = diagnostics.compute_global_temperature(ions)
    drift_hist[sample_idx] = diagnostics.compute_drift_velocity(ions)
    density_hist[sample_idx] = diagnostics.compute_density_profile(ions, z_grid, area)
    traj_hist[sample_idx] = ions.z[tracer_indices]
    sample_idx += 1

    # --- Main time loop -------------------------------------------------------

    for step in range(1, n_steps + 1):
        # Active particles
        N = ions.N
        z = ions.z[:N].copy()
        v = np.column_stack((ions.vx[:N], ions.vy[:N], ions.vz[:N]))
        q_arr = np.full(N, ions.q)
        m_arr = np.full(N, ions.m)

        # 1) Advance one step with Boris integrator (Lorentz force only)
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

        # 2) Magnetic mirror force: dv_parallel/dt = -(mu/m) dB/dz
        B_mag = B_mag_func(z)  # |B(z)|, shape (N,)
        dB_dz = dB_dz_func(z)  # dB/dz, shape (N,)

        v_perp2 = v[:, 0] ** 2 + v[:, 1] ** 2
        B_safe = B_mag + 1e-12  # avoid division by zero
        mu = 0.5 * ions.m * v_perp2 / B_safe

        dvz = -mirror_strength * (mu / ions.m) * dB_dz * dt
        v[:, 2] += dvz

        # 3) Write back to the Species arrays
        ions.z[:N] = z
        ions.vx[:N] = v[:, 0]
        ions.vy[:N] = v[:, 1]
        ions.vz[:N] = v[:, 2]

        # 4) Periodic BCs in z
        apply_periodic_boundaries(ions.z[:N], z_min, z_max)

        # 5) Record diagnostics
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

    # --- Final energy distribution -------------------------------------------

    energy_centers, energy_pdf = diagnostics.compute_energy_distribution(
        ions, n_bins=100, e_max=None
    )

    # --- Save results ---------------------------------------------------------

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
    from pathlib import Path

    run_magneticmirror_example()
    results_dir = Path(__file__).resolve().parent / "results"
