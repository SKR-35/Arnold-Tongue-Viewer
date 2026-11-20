"""Mathematical invariants of the circle-map lift and rotation estimator."""

import numpy as np
import pytest

from arnold_tongue_viewer.core import (
    CircleMapSettings,
    circle_map_step,
    rotation_number,
    rotation_number_grid,
)


@pytest.mark.parametrize("omega", [-0.25, 0.0, 0.125, 0.5, 1.0, 1.75])
def test_uncoupled_rotation_is_exact(omega: float) -> None:
    settings = CircleMapSettings(transient=5, samples=100)
    assert rotation_number(omega, 0.0, settings) == pytest.approx(omega, abs=1e-12)


def test_lift_commutes_with_integer_translation() -> None:
    theta, omega, coupling = 0.123, 0.37, 0.8
    assert circle_map_step(theta + 2, omega, coupling) == pytest.approx(
        circle_map_step(theta, omega, coupling) + 2
    )


def test_fixed_point_at_zero_for_zero_drive() -> None:
    assert rotation_number(0.0, 0.5, CircleMapSettings(samples=100)) == pytest.approx(0.0)


def test_half_locking_at_symmetry_point() -> None:
    # At theta0=0 and omega=1/2, phase alternates 0, 1/2 for all K.
    settings = CircleMapSettings(transient=10, samples=200, theta0=0)
    assert rotation_number(0.5, 0.8, settings) == pytest.approx(0.5, abs=1e-12)


def test_grid_matches_scalar_estimates() -> None:
    settings = CircleMapSettings(transient=20, samples=120)
    omega = np.array([[0.0, 0.25, 0.5], [0.1, 0.3, 0.7]])
    coupling = np.array([[0.0], [0.6]])
    result = rotation_number_grid(omega, coupling, settings)
    assert result.shape == (2, 3)
    for row in range(2):
        for col in range(3):
            assert result[row, col] == pytest.approx(
                rotation_number(float(omega[row, col]), float(coupling[row, 0]), settings),
                abs=1e-12,
            )


def test_broadcasting_and_zero_coupling() -> None:
    omega = np.array([0.1, 0.3, 0.9])
    result = rotation_number_grid(omega, np.array([[0.0], [0.0]]), CircleMapSettings(0, 20))
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, np.broadcast_to(omega, (2, 3)), atol=1e-12)


@pytest.mark.parametrize("transient,samples", [(-1, 10), (10, 0)])
def test_invalid_iteration_counts(transient: int, samples: int) -> None:
    with pytest.raises(ValueError):
        CircleMapSettings(transient=transient, samples=samples)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_parameters_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        rotation_number(bad, 0.5)
    with pytest.raises(ValueError):
        rotation_number_grid(np.array([0.2]), np.array([bad]))
