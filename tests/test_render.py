"""Rendering tests use short runs to remain fast in CI."""

import numpy as np
import pytest

from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.palettes import PALETTES, colorize
from arnold_tongue_viewer.render import Viewport, parameter_grid, render_rgb, rotation_field

SETTINGS = CircleMapSettings(transient=10, samples=40)


def test_grid_orientation_and_shape():
    omega, coupling = parameter_grid(Viewport(), 7, 5)
    assert omega.shape == coupling.shape == (5, 7)
    assert omega[0, 0] == pytest.approx(0.0)
    assert omega[-1, -1] == pytest.approx(1.0)
    assert coupling[0, 0] == pytest.approx(1.5)
    assert coupling[-1, 0] == pytest.approx(0.0)


def test_zero_coupling_has_analytic_rotation_number():
    view = Viewport(omega_min=0.1, omega_max=0.9, coupling_min=0.0, coupling_max=0.1)
    field = rotation_field(view, 9, 3, SETTINGS)
    np.testing.assert_allclose(field[-1], np.linspace(0.1, 0.9, 9), atol=1e-12)


@pytest.mark.parametrize("palette", list(PALETTES))
def test_render_shape_and_type(palette):
    rgb = render_rgb(Viewport(), 12, 8, SETTINGS, palette=palette)
    assert rgb.shape == (8, 12, 3)
    assert rgb.dtype == np.uint8


def test_render_is_deterministic():
    first = render_rgb(Viewport(), 8, 6, SETTINGS)
    second = render_rgb(Viewport(), 8, 6, SETTINGS)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("bounds", [(1, 1, 0, 1), (0, 1, 2, 1), (0, float("nan"), 0, 1)])
def test_invalid_viewport(bounds):
    with pytest.raises(ValueError):
        Viewport(*bounds)


@pytest.mark.parametrize("width,height", [(1, 5), (5, 1), (0, 0)])
def test_invalid_dimensions(width, height):
    with pytest.raises(ValueError):
        parameter_grid(Viewport(), width, height)


def test_invalid_palette_and_gamma():
    with pytest.raises(ValueError):
        colorize(np.zeros((2, 2)), "missing")
    with pytest.raises(ValueError):
        colorize(np.zeros((2, 2)), gamma=0)


def test_palette_endpoints():
    data = np.array([[0.0, 1.0]])
    rgb = colorize(data, "Grayscale")
    assert rgb.tolist() == [[[10, 10, 10], [242, 242, 242]]]
