"""
tests/test_fields.py

Comprehensive unit tests for fields.py.

- Basic shape/direction checks
- Analytic equivalence for B_mirror
- Derivative consistency with analytic dBz/dz
- Symmetry and scaling behavior
- Robustness to input types and edge cases
"""

from __future__ import annotations

import numpy as np

from test_particle_sim_1d import fields

# 1. Analytic equivalence for B_mirror


def test_B_mirror_matches_analytic():
    """Numerical B_mirror matches analytic formula exactly."""
    z = np.linspace(-0.05, 0.05, 200)
    B0, L = 1.2, 0.05
    B_num = fields.B_mirror(z, B0=B0, z_mirror=L)
    B_expected = B0 * (1 + (z / L) ** 2)
    np.testing.assert_allclose(B_num[:, 2], B_expected, rtol=1e-12)


# 2. Derivative (gradient) consistency


def test_B_mirror_derivative_consistency():
    """Numerical dBz/dz matches analytic derivative of B_mirror."""
    z = np.linspace(-0.05, 0.05, 400)
    B0, L = 1.0, 0.05
    Bz = fields.B_mirror(z, B0=B0, z_mirror=L)[:, 2]

    dBz_num = np.gradient(Bz, z)
    dBz_analytic = 2 * B0 * z / (L**2)

    # ignore the very edges and relax tolerance slightly
    np.testing.assert_allclose(dBz_num[5:-5], dBz_analytic[5:-5], rtol=5e-3, atol=1e-4)


# 3. Symmetry about z = 0


def test_B_mirror_symmetry():
    """B_mirror field must be perfectly symmetric: Bz(z) = Bz(-z)."""
    z = np.linspace(-0.05, 0.05, 1001)
    Bz = fields.B_mirror(z, B0=1.0, z_mirror=0.05)[:, 2]

    left = Bz[: len(Bz) // 2]
    right = Bz[len(Bz) // 2 + 1 :][::-1]
    np.testing.assert_allclose(left, right, rtol=1e-10)


# 4. Parameter-scaling behavior


def test_B_mirror_scales_with_B0():
    """Scaling B0 should scale the entire field linearly."""
    z = np.linspace(-0.05, 0.05, 50)
    B1 = fields.B_mirror(z, B0=1.0, z_mirror=0.05)[:, 2]
    B2 = fields.B_mirror(z, B0=2.0, z_mirror=0.05)[:, 2]
    np.testing.assert_allclose(B2, 2 * B1, rtol=1e-12)


def test_B_mirror_scales_with_zmirror():
    """Larger z_mirror should yield weaker mirror effect (flatter curve)."""
    z = np.linspace(-0.05, 0.05, 50)
    L1, L2 = 0.05, 0.1
    B1 = fields.B_mirror(z, B0=1.0, z_mirror=L1)[:, 2]
    B2 = fields.B_mirror(z, B0=1.0, z_mirror=L2)[:, 2]

    # Both fields equal at z=0
    np.testing.assert_allclose(B1[len(z) // 2], B2[len(z) // 2], atol=1e-3)
    # At edges, smaller L (stronger mirror) should give larger Bz
    assert B1[-1] > B2[-1]


# 5. Direction tests for uniform fields


def test_uniform_field_directions():
    """Uniform fields populate the correct axis only."""
    z = np.linspace(0, 1, 5)
    for func in [fields.E_uniform, fields.B_uniform]:
        for direction, idx in zip("xyz", [0, 1, 2], strict=False):
            F = func(z, 5.0, direction)
            assert F.shape == (5, 3)
            assert np.allclose(F[:, idx], 5.0)
            # All other components must be zero
            mask = [i for i in range(3) if i != idx]
            assert np.allclose(F[:, mask], 0.0)


# 6. Vector magnitude checks


def test_uniform_field_magnitude_constant():
    """Uniform field should have constant magnitude everywhere."""
    z = np.linspace(0, 1, 5)
    E = fields.E_uniform(z, E0=10.0, direction="x")
    magnitudes = np.linalg.norm(E, axis=1)
    assert np.allclose(magnitudes, 10.0)


# 7. Input type robustness


def test_accepts_python_lists():
    """Ensure field functions accept Python lists and return NumPy arrays."""
    z_list = [0.0, 0.5, 1.0]
    E = fields.E_uniform(z_list, E0=2.0)
    B = fields.B_uniform(z_list, B0=1.0)
    assert isinstance(E, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert E.shape == (3, 3)
    assert B.shape == (3, 3)


# 8. Edge case: empty input


def test_empty_input_returns_empty():
    """Functions should safely handle empty arrays."""
    E = fields.E_uniform(np.array([]), E0=5.0)
    B = fields.B_uniform(np.array([]), B0=1.0)
    assert E.shape == (0, 3)
    assert B.shape == (0, 3)
