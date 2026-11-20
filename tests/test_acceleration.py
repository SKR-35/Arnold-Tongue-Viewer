"""Cross-check accelerated and reference circle-map estimators."""

import numpy as np
import pytest

from arnold_tongue_viewer.core import (
    CircleMapSettings,
    accelerated_rotation_number_grid,
    rotation_number_grid,
)
from arnold_tongue_viewer.palettes import ORDER, PALETTES, colorize


def test_palette_order_and_membership():
    assert ORDER[:3] == ("Jade", "Yikique", "Pearl")
    assert len(ORDER) == 17
    assert tuple(PALETTES) == ORDER


@pytest.mark.parametrize("palette", ORDER)
def test_all_palettes_produce_rgb(palette):
    rgb = colorize(np.array([[0.0, 0.5, 1.0]]), palette)
    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize("coupling", [0.0, 0.3, 0.9, 1.4])
def test_accelerated_matches_numpy(coupling):
    omegas = np.linspace(0.05, 0.95, 12).reshape(3, 4)
    couplings = np.full_like(omegas, coupling)
    settings = CircleMapSettings(transient=12, samples=60)
    actual = accelerated_rotation_number_grid(omegas, couplings, settings)
    expected = rotation_number_grid(omegas, couplings, settings)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)


def test_accelerated_broadcast_and_nonfinite():
    omega = np.array([[0.1, 0.3, 0.5]])
    coupling = np.array([[0.0], [0.5]])
    result = accelerated_rotation_number_grid(omega, coupling, CircleMapSettings(5, 20))
    assert result.shape == (2, 3)
    with pytest.raises(ValueError):
        accelerated_rotation_number_grid(np.array([np.nan]), np.array([0.5]))
