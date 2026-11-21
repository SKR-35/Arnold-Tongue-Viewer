"""Reference gallery must remain deterministic and internally valid."""

import pytest

from arnold_tongue_viewer.map_families import MAP_FAMILIES
from arnold_tongue_viewer.palettes import PALETTES
from arnold_tongue_viewer.reference_gallery import REFERENCE_PRESETS, get_reference_preset


def test_five_unique_presets():
    assert len(REFERENCE_PRESETS) == 5
    assert len({p.label for p in REFERENCE_PRESETS}) == 5


@pytest.mark.parametrize("preset", REFERENCE_PRESETS, ids=lambda p: p.label)
def test_preset_is_valid(preset):
    assert preset.family in MAP_FAMILIES
    assert preset.palette == "Classic" or preset.palette in PALETTES
    assert preset.mode in {"Bifurcation", "Poincare recurrence", "Classic winding"}
    assert preset.view.omega_max > preset.view.omega_min
    assert preset.view.coupling_max > preset.view.coupling_min
    assert preset.samples > 0
    assert get_reference_preset(preset.label) is preset


def test_unknown_preset_rejected():
    with pytest.raises(ValueError, match="Unknown reference preset"):
        get_reference_preset("missing")
