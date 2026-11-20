"""Pure GUI navigation tests (no display required)."""

import pytest

from arnold_tongue_viewer.gui import pan_view, zoom_view
from arnold_tongue_viewer.render import Viewport


def test_zoom_at_center():
    view = zoom_view(Viewport(), 0.5, 0.5, 0.5)
    assert (view.omega_min, view.omega_max) == pytest.approx((0.25, 0.75))
    assert (view.coupling_min, view.coupling_max) == pytest.approx((0.375, 1.125))


def test_zoom_at_corner_keeps_anchor():
    view = zoom_view(Viewport(), 0.5, 0.0, 0.0)
    assert view.omega_min == 0
    assert view.coupling_max == 1.5


def test_pan_right_moves_view_left():
    view = pan_view(Viewport(), 0.2, 0)
    assert view.omega_min == pytest.approx(-0.2)


def test_pan_down_moves_view_up():
    view = pan_view(Viewport(), 0, 0.2)
    assert view.coupling_min == pytest.approx(0.3)


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_zoom(factor):
    with pytest.raises(ValueError):
        zoom_view(Viewport(), factor, 0.5, 0.5)


def test_reject_anchor_outside_canvas():
    with pytest.raises(ValueError):
        zoom_view(Viewport(), 0.5, 1.1, 0.5)
