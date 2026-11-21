import numpy as np
import pytest

from arnold_tongue_viewer.bifurcation import (
    BifurcationOptions,
    bifurcation_orbits,
    render_bifurcation,
)


@pytest.mark.parametrize("family", ["Sine", "Triangle", "Sawtooth"])
def test_bifurcation_shape_range_and_determinism(family):
    options = BifurcationOptions(transient=8, samples=15)
    first = bifurcation_orbits(12, options, family=family)
    assert first.shape == (15, 12)
    assert np.all((first >= 0) & (first < 1))
    assert np.array_equal(first, bifurcation_orbits(12, options, family=family))


def test_zero_coupling_orbit_matches_rigid_rotation():
    options = BifurcationOptions(omega=0.25, k_min=0, k_max=0.001, transient=0, samples=4, theta0=0)
    actual = bifurcation_orbits(2, options)
    assert np.allclose(actual[:, 0], [0.25, 0.5, 0.75, 0.0])


def test_render_dimensions_and_background():
    image = render_bifurcation(32, 24, BifurcationOptions(transient=5, samples=20))
    assert image.shape == (24, 32, 3)
    assert image.dtype == np.uint8
    assert np.any(image != image[0, 0])


@pytest.mark.parametrize(
    "kwargs", [{"k_min": 2, "k_max": 1}, {"samples": 0}, {"transient": -1}, {"omega": float("nan")}]
)
def test_reject_invalid_options(kwargs):
    with pytest.raises(ValueError):
        BifurcationOptions(**kwargs)


def test_reject_invalid_family_and_width():
    with pytest.raises(ValueError):
        bifurcation_orbits(2, BifurcationOptions(), family="Invalid")
    with pytest.raises(ValueError):
        bifurcation_orbits(1, BifurcationOptions())
