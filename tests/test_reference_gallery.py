"""Reference gallery must remain deterministic and internally valid."""

import pytest

from arnold_tongue_viewer.map_families import MAP_FAMILIES
from arnold_tongue_viewer.palettes import PALETTES
from arnold_tongue_viewer.reference_gallery import REFERENCE_PRESETS, get_reference_preset


def test_sixteen_unique_presets():
    assert len(REFERENCE_PRESETS) == 16
    assert len({p.label for p in REFERENCE_PRESETS}) == 16


@pytest.mark.parametrize("preset", REFERENCE_PRESETS, ids=lambda p: p.label)
def test_preset_is_valid(preset):
    assert preset.family in MAP_FAMILIES
    assert preset.palette == "Classic" or preset.palette in PALETTES
    assert preset.mode in {
        "Bifurcation",
        "Poincare recurrence",
        "Recurrence order q",
        "Classic winding",
    }
    assert preset.view.omega_max > preset.view.omega_min
    assert preset.view.coupling_max > preset.view.coupling_min
    assert preset.samples > 0
    assert get_reference_preset(preset.label) is preset


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="Unknown reference preset"):
        get_reference_preset("missing")


def test_startup_showcase_matches_gallery():
    showcase = get_reference_preset("06 · Rainbow Showcase · Classic winding")
    assert showcase.family == "Sine"
    assert showcase.mode == "Classic winding"
    assert showcase.palette == "Yikique"
    assert (showcase.transient, showcase.samples) == (250, 550)
    assert showcase.view == __import__(
        "arnold_tongue_viewer.render", fromlist=["Viewport"]
    ).Viewport(0, 1, 0, 3.0)


def test_new_presets_cover_distinct_observables_and_families():
    new = REFERENCE_PRESETS[10:]
    assert len(new) == 6
    assert {p.family for p in new} == {"Sine", "Triangle", "Sawtooth"}
    assert {p.mode for p in new} == {"Classic winding", "Poincare recurrence", "Recurrence order q"}
    assert len({p.palette for p in new}) == 6
    assert all(p.note for p in new)


def test_startup_preset_has_explicit_extended_view():
    from arnold_tongue_viewer.render import Viewport

    preset = get_reference_preset("06 · Rainbow Showcase · Classic winding")
    assert preset.view == Viewport(0, 1, 0, 3.0)
    assert preset.gamma == 0.9
