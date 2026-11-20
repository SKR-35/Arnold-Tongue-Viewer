import numpy as np
import pytest

from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.explorer import (
    EXPLORER_MODES,
    RecurrenceOptions,
    recurrence_field,
    render_explorer_rgb,
)
from arnold_tongue_viewer.render import Viewport


@pytest.mark.parametrize("family", ["Sine", "Triangle", "Sawtooth"])
def test_zero_coupling_half_turn_returns_in_two_steps(family):
    # Omega=1/2, K=0 -> phase returns after exactly two iterations.
    view = Viewport(0.5, 0.50000001, 0, 0.00000001)
    result = recurrence_field(
        view,
        2,
        2,
        CircleMapSettings(transient=4, samples=10),
        family=family,
        options=RecurrenceOptions(tolerance=0.01, max_steps=10),
    )
    assert np.all(result == 2)


@pytest.mark.parametrize("mode", EXPLORER_MODES)
@pytest.mark.parametrize("family", ["Sine", "Triangle", "Sawtooth"])
def test_render_modes_shape_and_dtype(mode, family):
    rgb = render_explorer_rgb(
        Viewport(),
        8,
        6,
        CircleMapSettings(transient=3, samples=15),
        mode=mode,
        family=family,
        recurrence=RecurrenceOptions(max_steps=15),
    )
    assert rgb.shape == (6, 8, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize("bad", [-0.1, 0, 0.5, float("nan")])
def test_invalid_tolerance(bad):
    with pytest.raises(ValueError):
        RecurrenceOptions(tolerance=bad)


def test_invalid_mode():
    with pytest.raises(ValueError, match="Unknown explorer"):
        render_explorer_rgb(Viewport(), 4, 4, mode="missing")
