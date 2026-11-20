import numpy as np
import pytest

from arnold_tongue_viewer.core import CircleMapSettings, circle_map_step, rotation_number_grid
from arnold_tongue_viewer.map_families import (
    MAP_FAMILIES,
    family_rotation_grid,
    forcing,
    map_step,
)


@pytest.mark.parametrize("family", MAP_FAMILIES)
def test_periodic_forcing(family):
    points = np.array([0.1, 0.3, 0.7, 0.9])
    assert np.allclose(forcing(points, family), forcing(points + 1, family))
    assert np.max(np.abs(forcing(points, family))) <= 1.0


@pytest.mark.parametrize("family", MAP_FAMILIES)
def test_zero_coupling_rotation(family):
    omega = np.array([[0.15, 0.3, 0.65]])
    coupling = np.zeros_like(omega)
    settings = CircleMapSettings(transient=10, samples=25)
    assert np.allclose(family_rotation_grid(omega, coupling, settings, family=family), omega)


def test_sine_matches_legacy_core():
    values = [0.0, 0.125, 0.4, 0.99]
    for x in values:
        assert map_step(x, 0.35, 0.8, "Sine") == pytest.approx(circle_map_step(x, 0.35, 0.8))
    omega = np.array([[0.2, 0.4], [0.6, 0.8]])
    coupling = np.array([[0.0, 0.2], [0.5, 1.1]])
    settings = CircleMapSettings(transient=20, samples=50)
    assert np.allclose(
        family_rotation_grid(omega, coupling, settings),
        rotation_number_grid(omega, coupling, settings),
        atol=1e-12,
    )


def test_forcing_shape_and_symmetry():
    x = np.array([0.125, 0.375, 0.625, 0.875])
    assert np.allclose(forcing(x, "Triangle"), [0.5, 0.5, -0.5, -0.5])
    assert np.allclose(forcing(x, "Sawtooth"), [0.25, 0.75, -0.75, -0.25])


@pytest.mark.parametrize("family", MAP_FAMILIES)
def test_finite_grid(family):
    settings = CircleMapSettings(transient=5, samples=12)
    grid = family_rotation_grid(
        np.array([[0.1], [0.4]]), np.array([[0.0, 0.3, 0.8]]), settings, family=family
    )
    assert grid.shape == (2, 3)
    assert np.isfinite(grid).all()


def test_invalid_family_and_parameters():
    with pytest.raises(ValueError, match="Unknown"):
        forcing(0.2, "Unknown")
    with pytest.raises(ValueError, match="finite"):
        family_rotation_grid(np.array([np.nan]), np.array([0.1]))
