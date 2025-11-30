"""
test_physics_integration.py

Integration tests for the collision physics and diagnostics modules.
Combines thermalization, isotropization, and density profile verification.

Tests
-----
test_isothermal_relaxation   : Verifies T_ion decays to T_neutral.
test_velocity_isotropization : Verifies energy redistribution (Tx -> Ty, Tz).
test_distribution_relaxation : Verifies evolution to Maxwellian energy distribution.
test_density_profile_reconstruction : Verifies density profile stability under collisions.
"""

from __future__ import annotations

import numpy as np

from test_particle_sim_1d import collisions, diagnostics
from test_particle_sim_1d.initialization import constants
from test_particle_sim_1d.particles import Species


def constant_sigma(g: np.ndarray) -> np.ndarray:
    """Helper: Constant cross-section of 1e-19 m^2."""
    return np.full_like(g, 1.0e-19)


def test_isothermal_relaxation():
    """
    Verify that hot ions relax to the neutral gas temperature over time.
    Case: 0D Box (Fields=0), Hot Ions -> Cold Neutrals.
    """
    # --- 1. Simulation Parameters ---
    n_particles = 20000
    dt = 1e-8
    n_steps = 1000

    # Physics Parameters
    T_ion_init_eV = 1.0  # Start hot (~11600 K)
    T_neutral_K = 300.0  # Background is cool
    neutral_density = 2e21

    # --- 2. Initialize Species ---
    ions = Species(
        q=constants.q_e, m=constants.m_p, name="proton", capacity=n_particles
    )
    ions.initialize_maxwellian(
        n=n_particles, z_min=0.0, z_max=0.1, temperature={"eV": T_ion_init_eV}, seed=42
    )

    T_start = diagnostics.compute_global_temperature(ions)
    print(f"\n[Thermalization] Start T_ion: {T_start:.2f} K (Target: {T_neutral_K} K)")

    assert T_start > 10000, "Initial temperature should be high."

    # --- 3. Run Time Loop ---
    temp_history = [T_start]

    for _ in range(n_steps):
        collisions.apply_elastic_collisions(
            species=ions,
            neutral_density=neutral_density,
            neutral_temp_K=T_neutral_K,
            neutral_mass=constants.m_p,
            cross_section_func=constant_sigma,
            dt=dt,
            seed=None,
        )
        current_T = diagnostics.compute_global_temperature(ions)
        temp_history.append(current_T)

    # --- 4. Verify Results ---
    T_final = temp_history[-1]
    print(f"[Thermalization] End   T_ion: {T_final:.2f} K")

    assert T_final < T_start, "Ions should have cooled down."

    # Check convergence to neutral temp (within 20% margin for noise)
    assert abs(T_final - T_neutral_K) < 0.2 * T_neutral_K, (
        f"Final T {T_final:.1f} K not close to {T_neutral_K} K"
    )

    # Check monotonicity (compare start avg vs end avg)
    avg_start = np.mean(temp_history[:10])
    avg_end = np.mean(temp_history[-10:])
    assert avg_end < avg_start


def test_velocity_isotropization():
    """
    Verify that energy transfers from hot dimensions to cold dimensions.
    Case: Hot in X, Cold in Y/Z -> Equipartition.
    """
    # --- 1. Parameters ---
    n_particles = 10000
    dt = 1e-8
    n_steps = 500

    T_neutral_K = 300.0
    neutral_density = 2e21

    # --- 2. Initialize Anisotropic Species ---
    ions = Species(
        q=constants.q_e, m=constants.m_p, name="proton", capacity=n_particles
    )

    # Manually create anisotropic distribution
    rng = np.random.default_rng(42)
    v_th_hot = 20000.0  # Hot in X
    v_th_cold = 0.0  # Cold in Y, Z

    vx = rng.normal(0.0, v_th_hot, n_particles)
    vy = rng.normal(0.0, v_th_cold, n_particles)
    vz = rng.normal(0.0, v_th_cold, n_particles)
    z = np.zeros(n_particles)

    ions.add_particles(z, np.column_stack((vx, vy, vz)))

    # Use diagnostics module for component temps
    Tx_start, Ty_start, Tz_start = diagnostics.compute_component_temperatures(ions)
    print(
        f"\n[Isotropization] Start Tx: {Tx_start:.0f}, Ty: {Ty_start:.0f}, Tz: {Tz_start:.0f} K"
    )

    assert Tx_start > 10000
    assert Ty_start < 10

    # --- 3. Run Collision Loop ---
    for _ in range(n_steps):
        collisions.apply_elastic_collisions(
            species=ions,
            neutral_density=neutral_density,
            neutral_temp_K=T_neutral_K,
            neutral_mass=constants.m_p,
            cross_section_func=constant_sigma,
            dt=dt,
        )

    # --- 4. Verify Results ---
    Tx_final, Ty_final, Tz_final = diagnostics.compute_component_temperatures(ions)
    print(
        f"[Isotropization] End   Tx: {Tx_final:.0f}, Ty: {Ty_final:.0f}, Tz: {Tz_final:.0f} K"
    )

    # Y and Z should heat up, X should cool down
    assert Ty_final > Ty_start
    assert Tz_final > Tz_start
    assert Tx_final < Tx_start

    # Check for Equipartition (Isotropy)
    T_mean = (Tx_final + Ty_final + Tz_final) / 3.0

    assert abs(Tx_final - T_mean) < 0.2 * T_mean, "Tx did not converge to mean"
    assert abs(Ty_final - T_mean) < 0.2 * T_mean, "Ty did not converge to mean"
    assert abs(Tz_final - T_mean) < 0.2 * T_mean, "Tz did not converge to mean"


def test_distribution_relaxation():
    """
    Verify that a mono-energetic beam relaxes to a Maxwellian distribution.
    Tests: collisions, compute_temperature_profile, compute_energy_distribution.
    """
    # --- 1. Parameters ---
    n_particles = 20000
    dt = 1e-8
    n_steps = 1000

    T_neutral_K = 11600.0  # ~1 eV background
    T_neutral_eV = 1.0
    neutral_density = 2e21

    # --- 2. Initialize Mono-energetic Species ---
    ions = Species(
        q=constants.q_e, m=constants.m_p, name="proton", capacity=n_particles
    )

    # Initialize uniform positions
    ions.initialize_uniform(
        n_particles, z_min=0.0, z_max=0.1, v0=[0.0, 0.0, 0.0], seed=42
    )

    # Overwrite velocities to be a "Cold Beam"
    energy_init_eV = 5.0
    v_mag = np.sqrt(2 * energy_init_eV * constants.q_e / constants.m_p)

    rng = np.random.default_rng(42)
    phi = rng.uniform(0, 2 * np.pi, n_particles)
    cos_theta = rng.uniform(-1, 1, n_particles)
    sin_theta = np.sqrt(1 - cos_theta**2)

    ions.vx[:n_particles] = v_mag * sin_theta * np.cos(phi)
    ions.vy[:n_particles] = v_mag * sin_theta * np.sin(phi)
    ions.vz[:n_particles] = v_mag * cos_theta

    print(f"\n[Distribution] Start: Mono-energetic at {energy_init_eV} eV")

    # --- 3. Run Collision Loop ---
    for _ in range(n_steps):
        collisions.apply_elastic_collisions(
            species=ions,
            neutral_density=neutral_density,
            neutral_temp_K=T_neutral_K,
            neutral_mass=constants.m_p,
            cross_section_func=constant_sigma,
            dt=dt,
        )

    # --- 4. Verify Temperature Profile ---
    z_grid = np.linspace(0.0, 0.1, 11)
    T_profile = diagnostics.compute_temperature_profile(ions, z_grid)
    avg_profile_T = np.mean(T_profile)
    print(
        f"[Distribution] End Avg Profile T: {avg_profile_T:.0f} K (Target: {T_neutral_K:.0f})"
    )

    assert abs(avg_profile_T - T_neutral_K) < 0.2 * T_neutral_K, (
        "Temperature profile average did not converge"
    )

    # --- 5. Verify Energy Distribution (IEDF) ---
    # Using 100 bins as requested
    centers, pdf = diagnostics.compute_energy_distribution(ions, n_bins=100, e_max=5.0)

    kT = T_neutral_eV
    theory_pdf = (
        2 * np.sqrt(centers / np.pi) * (1 / kT) ** (1.5) * np.exp(-centers / kT)
    )

    # --- FIXED L1 NORM CALCULATION ---
    # 1. Calculate the bin width (Delta E)
    bin_width = centers[1] - centers[0]

    # 2. Calculate sum of absolute differences
    diff = np.abs(pdf[1:] - theory_pdf[1:])
    sum_diff = np.sum(diff)

    # 3. Calculate Area Error (Integral difference)
    # This value represents the total probability mass that is "misplaced"
    l1_area_error = sum_diff * bin_width

    print(f"[Distribution] L1 Area Error: {l1_area_error:.3f}")

    # Threshold: 0.15 is generally safe for N=20000 particles and 100 bins.
    assert l1_area_error < 0.15, (
        f"L1 Area Error {l1_area_error:.3f} is too high; distribution shape is not Maxwellian."
    )


def test_density_profile_reconstruction():
    """
    Verify that compute_density_profile correctly measures a Step-Function density
    AND that the Collision Module can act on this non-uniform distribution.

    Case: Step Function Density (2:1 Contrast) + Collisional Heating (Cold -> Hot).
    """
    # --- 1. Parameters ---
    n_particles = 30000
    L = 0.1
    half_L = L / 2
    n_bins = 10
    area = 0.01

    # Collision Parameters
    dt = 1e-8
    n_steps = 200
    T_neutral_K = 1000.0  # Hot neutrals to heat our cold ions
    neutral_density = 2e21

    # --- 2. Initialize Species ---
    ions = Species(
        q=constants.q_e, m=constants.m_p, name="proton", capacity=n_particles
    )

    # Create Step Function Distribution manually
    rng = np.random.default_rng(42)
    n_left = 20000
    n_right = 10000

    z_left = rng.uniform(0.0, half_L, n_left)
    z_right = rng.uniform(half_L, L, n_right)
    z_all = np.concatenate((z_left, z_right))

    # Initialize velocities to ZERO (Cold)
    v_zeros = np.zeros((n_particles, 3))

    ions.add_particles(z_all, v_zeros)

    # Check Initial Temperature (Should be 0)
    T_init = diagnostics.compute_global_temperature(ions)
    print(f"\n[Density+Collisions] Start T: {T_init:.1f} K")
    assert T_init == 0.0, "Initial temperature should be 0"

    # --- 3. Run Collision Loop (Heating) ---
    # We do NOT update positions (z), only velocities via collisions.
    # This ensures density profile should remain constant while T rises.

    for _ in range(n_steps):
        collisions.apply_elastic_collisions(
            species=ions,
            neutral_density=neutral_density,
            neutral_temp_K=T_neutral_K,
            neutral_mass=constants.m_p,
            cross_section_func=constant_sigma,
            dt=dt,
        )

    # --- 4. Verify Collision Physics (Heating) ---
    T_final = diagnostics.compute_global_temperature(ions)
    print(f"[Density+Collisions] End T:   {T_final:.1f} K (Expected > 50 K)")

    # Particles should have heated up significantly from 0 K
    assert T_final > 50.0, "Collisions failed to heat the particles"

    # --- 5. Verify Density Profile (Diagnostics) ---
    # The density profile must remain exactly the step function, verifying that
    # collisions modified velocities without corrupting positions.

    z_grid = np.linspace(0.0, L, n_bins + 1)
    dz = L / n_bins
    bin_volume = dz * area

    density_profile = diagnostics.compute_density_profile(ions, z_grid, area=area)

    expected_density_left = (n_left / 5.0) / bin_volume
    expected_density_right = (n_right / 5.0) / bin_volume

    avg_density_left = np.mean(density_profile[:5])
    avg_density_right = np.mean(density_profile[5:])

    print(f"[Density+Collisions] Density Left:  {avg_density_left:.2e}")
    print(f"[Density+Collisions] Density Right: {avg_density_right:.2e}")

    # Verify Accuracy
    assert abs(avg_density_left - expected_density_left) < 0.05 * expected_density_left
    assert (
        abs(avg_density_right - expected_density_right) < 0.05 * expected_density_right
    )

    # Verify Contrast (2:1 ratio)
    assert avg_density_left > 1.8 * avg_density_right, "Density contrast not preserved"


if __name__ == "__main__":
    try:
        test_isothermal_relaxation()
        test_velocity_isotropization()
        test_distribution_relaxation()
        test_density_profile_reconstruction()
        print("\nAll Physics Tests Passed!")
    except AssertionError as e:
        print(f"\nTest Failed: {e}")
