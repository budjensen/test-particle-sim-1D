"""
uniformE_plotting.py

Combined loader + plotting for the uniformE simulation.

It:
1. Loads uniformE_results.npz from the ../results directory.
2. Generates:
   - Temperature history
   - Energy distribution
   - Density profile (final snapshot)
   - Tracer particle trajectories

All plots include titles, axis labels with units, and legends.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Plotting helper functions


def plot_temperature_history(time_s, temperature_K, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(time_s, temperature_K, label="Temperature", linewidth=2)

    plt.title("Global Temperature vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.grid(True)
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_energy_distribution(energy_eV, pdf, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(energy_eV, pdf, label="Energy PDF", linewidth=2)

    plt.title("Energy Distribution Function")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Probability Density (1/eV)")
    plt.grid(True)
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_density_profile(z_centers, density_m3, time_label="", save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(z_centers, density_m3, label=f"Density {time_label}", linewidth=2)

    plt.title("Density Profile")
    plt.xlabel("Position z (m)")
    plt.ylabel("Density (m⁻³)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.ylim(0, 5.0e5)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_trajectory(time_s, trajectories, save_path=None):
    """
    Plot tracer particle trajectories with fading (light→dark blue)
    and hide jumps caused by periodic wrapping.
    """
    plt.figure(figsize=(7, 5))

    n_tracers = trajectories.shape[1]
    N = len(time_s)

    # Fading alpha values from light → dark blue
    alphas = np.linspace(0.1, 1.0, N)

    # Estimate domain length to detect wrap jumps
    L = trajectories.max() - trajectories.min()
    jump_threshold = 0.5 * L

    for i in range(n_tracers):
        z = trajectories[:, i]
        t = time_s

        # Identify continuous segments (no wrap jumps)
        segment_starts = [0]
        for k in range(1, N):
            if abs(z[k] - z[k - 1]) > jump_threshold:
                segment_starts.append(k)
        segment_starts.append(N)

        # Plot segments with fading
        first_segment = True
        for start, end in pairwise(segment_starts):
            if end - start < 2:
                continue

            # Draw each small segment with fading alpha
            for k in range(start, end - 1):
                plt.plot(
                    t[k : k + 2],
                    z[k : k + 2],
                    color="tab:blue",
                    alpha=alphas[k],  # ← fade light → dark
                    linewidth=1.5,
                    label=(f"Tracer {i}" if first_segment and k == start else None),
                )

            first_segment = False

    plt.title("Tracer Particle Trajectories")
    plt.xlabel("Time (s)")
    plt.ylabel("Position z (m)")
    plt.grid(True)
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Main: load results and make plots


def main():
    # Directory layout:
    # examples/
    #   example_uniformE/
    #       uniformE.py
    #       results/
    #           uniformE_results.npz
    #       plotting/
    #           uniformE_plotting.py  <-- this file
    base_dir = Path(__file__).resolve().parent  # .../uniformE/plotting
    example_dir = base_dir.parent  # .../uniformE
    results_dir = example_dir / "results"

    # Save PNGs under the existing 'plotting' folder
    plots_dir = base_dir

    results_path = results_dir / "uniformE_results.npz"

    if not results_path.is_file():
        msg = (
            f"Could not find results file at: {results_path}\n"
            "Make sure you've run uniformE.py first."
        )
        raise FileNotFoundError(msg)

    # Load results
    data = np.load(results_path)

    time = data["time"]
    temperature = data["temperature"]
    z_grid = data["z_grid"]  # bin edges
    density = data["density_profile"]  # shape (N_samples, N_bins)
    tracer_traj = data["tracer_trajectories"]
    energy_centers = data["energy_centers"]
    energy_pdf = data["energy_pdf"]

    # Compute bin centers from edges
    z_centers = 0.5 * (z_grid[:-1] + z_grid[1:])

    # 1. Temperature vs time
    plot_temperature_history(
        time,
        temperature,
        save_path=plots_dir / "temperature_history.png",
    )

    # 2. Energy distribution
    plot_energy_distribution(
        energy_centers,
        energy_pdf,
        save_path=plots_dir / "energy_distribution.png",
    )

    # 3. Final density profile
    plot_density_profile(
        z_centers,
        density[-1],
        time_label=f"(t = {time[-1]:.3e} s)",
        save_path=plots_dir / "density_profile_final.png",
    )

    # 4. Tracer trajectories
    plot_trajectory(
        time,
        tracer_traj,
        save_path=plots_dir / "tracer_trajectories.png",
    )


if __name__ == "__main__":
    main()
