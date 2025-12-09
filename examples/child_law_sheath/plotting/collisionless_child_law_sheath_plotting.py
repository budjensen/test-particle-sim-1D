"""
child_law_sheath_plotting.py

Combined loader + plotting for the Child law sheath example.

It:
1. Loads child_sheath_results.npz from the ../results directory.
2. Generates:
   - Energy distribution
   - Tracer particle trajectories

All plots include titles, axis labels with units, and legends.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(time_s, trajectories, save_path=None):
    """
    Plot tracer particle trajectories.

    Parameters
    ----------
    time_s : np.ndarray
        Time array of shape (N_time,).
    trajectories : np.ndarray
        z positions of tracers, shape (N_time, N_tracers).
    """
    plt.figure(figsize=(7, 5))

    n_tracers = trajectories.shape[1]
    for i in range(n_tracers):
        plt.plot(time_s, trajectories[:, i], label=f"Tracer {i}", linewidth=1.5)

    plt.title("Tracer Particle Trajectories: Collisionless Child Law Sheath")
    plt.xlabel("Time (s)")
    plt.ylabel("Position z (m)")
    plt.grid(True)
    plt.legend(loc="lower right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_energy_distribution(energy_eV, pdf, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(energy_eV, pdf, label="Energy PDF", linewidth=2)

    plt.title("Surface IEDF: Collisionless Child Law Sheath")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Probability Density (1/eV)")
    plt.grid(True)
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # Load results
    results_path = (
        Path(__file__).parent.parent
        / "collisionless_results"
        / "child_sheath_results.npz"
    )
    data = np.load(results_path)

    time = data["time"]
    tracer_traj = data["tracer_trajectories"]
    energy_centers = data["energy_centers"]
    energy_pdf = data["energy_pdf"]

    plot_energy_distribution(
        energy_eV=energy_centers,
        pdf=energy_pdf,
        save_path=Path(__file__).parent.parent
        / "plotting/collisionless_child_sheath_energy_distribution.png",
    )
    plot_trajectory(
        time,
        tracer_traj,
        save_path=Path(__file__).parent.parent
        / "plotting/collisionless_child_sheath_tracer_trajectories.png",
    )


if __name__ == "__main__":
    main()
