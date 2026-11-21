import numpy as np
import pytest

from arnold_tongue_viewer.bifurcation import (
    BifurcationOptions,
    bifurcation_density,
    render_bifurcation,
)


def test_zoomed_phase_density():
    opts = BifurcationOptions(
        k_min=1.0, k_max=2.0, phase_min=0.2, phase_max=0.8, transient=10, samples=20
    )
    density = bifurcation_density(32, 24, opts)
    assert density.shape == (24, 32)
    assert np.isfinite(density).all()
    assert np.all((density >= 0) & (density <= 1))


@pytest.mark.parametrize("palette", ["Classic", "Yikique", "plasma"])
def test_palettes(palette):
    rgb = render_bifurcation(24, 18, BifurcationOptions(transient=5, samples=10), palette=palette)
    assert rgb.shape == (18, 24, 3)
    assert rgb.dtype == np.uint8


@pytest.mark.parametrize("bounds", [(-0.1, 0.8), (0.3, 1.1), (0.8, 0.3)])
def test_invalid_phase_limits(bounds):
    with pytest.raises(ValueError, match="Phase"):
        BifurcationOptions(phase_min=bounds[0], phase_max=bounds[1])
