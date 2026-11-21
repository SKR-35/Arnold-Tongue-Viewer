"""Reproducible, reference-inspired exploration presets.

These are NOT pixel-exact reconstructions of the cited artwork. The original
rendering algorithms and coordinate conventions may differ from ours.
"""

from __future__ import annotations

from dataclasses import dataclass

from .explorer import RecurrenceOptions
from .render import Viewport


@dataclass(frozen=True, slots=True)
class ReferencePreset:
    label: str
    family: str
    mode: str
    palette: str
    view: Viewport
    transient: int
    samples: int
    gamma: float = 1.0
    recurrence: RecurrenceOptions = RecurrenceOptions()
    note: str = ""


REFERENCE_PRESETS = (
    ReferencePreset(
        "01 · Circle-map bifurcation",
        "Sine",
        "Bifurcation",
        "Classic",
        Viewport(),
        400,
        240,
        note="Open the bifurcation explorer; try Omega 0.35 and K 0–4.",
    ),
    ReferencePreset(
        "02 · recur-circ",
        "Sine",
        "Poincare recurrence",
        "turbo",
        Viewport(0, 1, 0, 2.4),
        150,
        300,
        0.85,
        RecurrenceOptions(0.0075, 450),
        "First-return-time visualization for the smooth sine circle map.",
    ),
    ReferencePreset(
        "03 · recur-tri",
        "Triangle",
        "Poincare recurrence",
        "turbo",
        Viewport(0, 1, 0, 3.0),
        120,
        280,
        0.9,
        RecurrenceOptions(0.006, 400),
        "First-return-time visualization for piecewise-linear triangle forcing.",
    ),
    ReferencePreset(
        "04 · recur-tri-q",
        "Triangle",
        "Poincare recurrence",
        "turbo",
        Viewport(0.25, 0.75, 0.4, 3.5),
        200,
        350,
        0.8,
        RecurrenceOptions(0.0025, 600),
        "Exploratory zoom with tighter recurrence threshold; not the original q observable.",
    ),
    ReferencePreset(
        "05 · wind-circ",
        "Sine",
        "Classic winding",
        "turbo",
        Viewport(0, 1, 0, 3.0),
        250,
        550,
        0.9,
        note="Cyclic color bands of estimated winding; not a replica of source coloring.",
    ),
    ReferencePreset(
        "06 · Rainbow Showcase · Classic winding",
        "Sine",
        "Classic winding",
        "turbo",
        Viewport(0, 1, 0, 3.0),
        250,
        550,
        0.9,
        note="Rainbow winding overview; also the default startup view.",
    ),
    ReferencePreset(
        "07 · Rainbow winding · Close-up",
        "Sine",
        "Classic winding",
        "turbo",
        Viewport(0.22, 0.78, 0.5, 2.7),
        250,
        650,
        0.9,
        note="Explore nested winding structures near the center of the parameter plane.",
    ),
    ReferencePreset(
        "08 · Triangle · Recurrence detail",
        "Triangle",
        "Poincare recurrence",
        "turbo",
        Viewport(0.1, 0.9, 0.7, 2.8),
        250,
        500,
        0.85,
        RecurrenceOptions(0.004, 600),
        "Higher-sample triangular forcing recurrence; exploratory preset.",
    ),
    ReferencePreset(
        "09 · Sine · Recurrence close-up",
        "Sine",
        "Poincare recurrence",
        "Yikique",
        Viewport(0.25, 0.75, 0.2, 2.3),
        200,
        450,
        0.85,
        RecurrenceOptions(0.005, 500),
        "Fine-grained first-return structures with a contrasting palette.",
    ),
    ReferencePreset(
        "10 · Sawtooth · Winding bands",
        "Sawtooth",
        "Classic winding",
        "turbo",
        Viewport(0, 1, 0, 3.0),
        250,
        550,
        0.9,
        note="Explore the discontinuous sawtooth map's winding structures.",
    ),
)


def get_reference_preset(label: str) -> ReferencePreset:
    for preset in REFERENCE_PRESETS:
        if preset.label == label:
            return preset
    raise ValueError(f"Unknown reference preset: {label}")
