"""Navigation tests for rectangular selection."""

import pytest

from arnold_tongue_viewer.gui import rectangle_view
from arnold_tongue_viewer.render import Viewport


def test_full_rectangle_preserves_view():
    view = Viewport()
    assert rectangle_view(view, 0, 0, 1, 1) == view


def test_center_rectangle():
    view = rectangle_view(Viewport(), 0.25, 0.25, 0.75, 0.75)
    assert view.omega_min == pytest.approx(0.25)
    assert view.omega_max == pytest.approx(0.75)
    assert view.coupling_min == pytest.approx(0.375)
    assert view.coupling_max == pytest.approx(1.125)


def test_reverse_drag_same_result():
    view = Viewport()
    assert rectangle_view(view, 0.8, 0.7, 0.2, 0.1) == rectangle_view(view, 0.2, 0.1, 0.8, 0.7)


@pytest.mark.parametrize("coords", [(0, 0, 0, 1), (0, 0, 1, 0), (-0.1, 0, 1, 1), (0, 0, 2, 1)])
def test_invalid_rectangle(coords):
    with pytest.raises(ValueError):
        rectangle_view(Viewport(), *coords)
