"""Color and recurrence-order regressions."""

import numpy as np

from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.explorer import RecurrenceOptions, render_explorer_rgb
from arnold_tongue_viewer.palettes import PALETTES, colorize
from arnold_tongue_viewer.render import Viewport


def test_yikique_is_vivid_and_distinct_from_turbo():
    stops = np.asarray(PALETTES["Yikique"])
    assert len(stops) >= 10
    assert np.ptp(stops, axis=0).min() > 180
    assert not np.array_equal(
        colorize(np.array([[0.25, 0.5, 0.75]]), "Yikique"),
        colorize(np.array([[0.25, 0.5, 0.75]]), "turbo"),
    )


def test_q_mode_is_deterministic_and_rgb():
    kw = dict(
        view=Viewport(0.2, 0.8, 0.2, 1.6),
        width=14,
        height=9,
        settings=CircleMapSettings(transient=8, samples=15),
        family="Triangle",
        mode="Recurrence order q",
        palette="Yikique",
        recurrence=RecurrenceOptions(0.01, 35),
    )
    first = render_explorer_rgb(**kw)
    assert first.shape == (9, 14, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, render_explorer_rgb(**kw))
