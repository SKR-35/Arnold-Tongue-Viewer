"""Suggested export names are deterministic and contain relevant settings."""

import pytest

from arnold_tongue_viewer.export import suggested_filename


def name(**overrides):
    settings = dict(
        mode="Mode locking",
        palette="Yikique",
        gamma=1.0,
        transient=100,
        samples=250,
        omega_min=0.0,
        omega_max=1.0,
        coupling_min=0.0,
        coupling_max=1.5,
    )
    settings.update(overrides)
    return suggested_filename(**settings)


def test_locking_name_excludes_gamma():
    assert name() == ("arnold_mode-locking_yikique_t100_s250_O0.0000-1.0000_K0.0000-1.5000.png")
    assert name(gamma=2.0) == name()


def test_rotation_name_includes_gamma():
    assert "_g1.25_" in name(mode="Rotation number", gamma=1.25)


def test_filename_changes_with_viewport():
    assert name(omega_min=0.25) != name()


def test_extension():
    assert name(extension=".jpg").endswith(".jpg")


@pytest.mark.parametrize(
    "changes",
    [
        {"samples": 0},
        {"transient": -1},
        {"omega_min": 2.0},
        {"coupling_max": float("nan")},
        {"extension": ".gif"},
        {"palette": "!!!"},
        {"mode": "Rotation number", "gamma": 0},
    ],
)
def test_invalid_values(changes):
    with pytest.raises(ValueError):
        name(**changes)
