"""Numerical locking diagnostics and display tests."""

import numpy as np
import pytest

from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.locking import colorize_locking, locking_field, rational_labels
from arnold_tongue_viewer.render import Viewport


def test_rational_labels_reduced():
    labels = rational_labels(4)
    assert (1, 2) in labels
    assert (2, 4) not in labels
    assert (0, 1) in labels


def test_invalid_denominator():
    with pytest.raises(ValueError):
        rational_labels(0)


def test_locking_zero_tongue():
    # Omega=0, K=0.5 has an attracting fixed point at phase zero.
    field = locking_field(
        Viewport(0, 0.01, 0.5, 0.51), 3, 3, CircleMapSettings(transient=100, samples=100)
    )
    assert field[0, 0] == 101


def test_unlocked_at_zero_coupling():
    field = locking_field(
        Viewport(0.23, 0.24, 0.0, 0.00001), 3, 3, CircleMapSettings(transient=50, samples=50)
    )
    assert field[2, 0] == 0


def test_locking_color_shape():
    labels = np.array([[0, 101], [203, 0]], dtype=np.int16)
    rgb = colorize_locking(labels)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[0, 0], [10, 12, 25])
    assert not np.array_equal(rgb[0, 1], rgb[1, 0])


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        locking_field(Viewport(), 2, 2, tolerance=-1)
