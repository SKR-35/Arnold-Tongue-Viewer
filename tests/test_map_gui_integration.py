import numpy as np
import pytest

from arnold_tongue_viewer.animation import render_animation_frame
from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.map_families import family_rotation_grid
from arnold_tongue_viewer.render import Viewport, parameter_grid, render_rgb, rotation_field


@pytest.mark.parametrize("family", ["Sine", "Triangle", "Sawtooth"])
def test_rotation_field_uses_selected_family(family):
    view = Viewport()
    settings = CircleMapSettings(transient=8, samples=16)
    omega, coupling = parameter_grid(view, 8, 6)
    expected = family_rotation_grid(omega, coupling, settings, family=family)
    assert np.allclose(rotation_field(view, 8, 6, settings, family=family), expected)


@pytest.mark.parametrize("family", ["Sine", "Triangle", "Sawtooth"])
def test_render_rgb_and_gif_frame_support_family(family):
    settings = CircleMapSettings(transient=5, samples=12)
    rgb = render_rgb(Viewport(), 8, 6, settings, palette="Yikique", family=family)
    assert rgb.shape == (6, 8, 3)
    image = render_animation_frame(
        Viewport(),
        settings,
        width=8,
        height=6,
        palette="Yikique",
        mode="Rotation number",
        gamma=1.0,
        family=family,
    )
    assert image.size == (8, 6)


def test_non_sine_locking_rejected():
    with pytest.raises(ValueError, match="only the Sine"):
        render_animation_frame(
            Viewport(),
            CircleMapSettings(),
            width=8,
            height=6,
            palette="Yikique",
            mode="Mode locking",
            gamma=1.0,
            family="Triangle",
        )


def test_invalid_family_rejected():
    with pytest.raises(ValueError, match="Unknown map family"):
        rotation_field(Viewport(), 8, 6, family="Not a map")
