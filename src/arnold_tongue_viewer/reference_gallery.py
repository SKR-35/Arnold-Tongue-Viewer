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
        "04 · recur-tri-q (exploratory)",
        "Triangle",
        "Recurrence order q",
        "Yikique",
        Viewport(0.25, 0.75, 0.4, 3.5),
        200,
        350,
        0.8,
        RecurrenceOptions(0.0025, 600),
        "Discrete observed return-step q proxy; original reference q observable remains unverified.",
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
        "Yikique",
        Viewport(0, 1, 0, 3.0),
        250,
        550,
        0.9,
        note="Vivid Yikique winding overview; also the default startup view.",
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
    ReferencePreset(
        "11 · Resonance microscope",
        "Sine",
        "Classic winding",
        "magma",
        Viewport(0.29, 0.42, 0.05, 1.05),
        350,
        850,
        0.78,
        note="Narrow resonance window with a dark, high-contrast palette.",
    ),
    ReferencePreset(
        "12 · Triangle return mosaic",
        "Triangle",
        "Recurrence order q",
        "twilight",
        Viewport(0.05, 0.95, 0.1, 2.2),
        240,
        460,
        0.72,
        RecurrenceOptions(0.003, 750),
        "Discrete first-return classes for triangle forcing; not a proven period classification.",
    ),
    ReferencePreset(
        "13 · Sawtooth return strata",
        "Sawtooth",
        "Poincare recurrence",
        "inferno",
        Viewport(0.05, 0.95, 0.2, 2.6),
        220,
        400,
        0.68,
        RecurrenceOptions(0.008, 400),
        "Return-time bands for discontinuous sawtooth forcing.",
    ),
    ReferencePreset(
        "14 · High-coupling kaleidoscope",
        "Sine",
        "Classic winding",
        "plasma",
        Viewport(0.12, 0.88, 2.2, 5.2),
        300,
        750,
        0.88,
        note="Explore the high-coupling regime beyond the initial overview.",
    ),
    ReferencePreset(
        "15 · Rational island scan",
        "Sine",
        "Recurrence order q",
        "cividis",
        Viewport(0.18, 0.53, 0.15, 1.8),
        260,
        520,
        0.82,
        RecurrenceOptions(0.004, 900),
        "A focused view of observed return-step classes around low-coupling resonances.",
    ),
    ReferencePreset(
        "16 · Sawtooth spectral bands",
        "Sawtooth",
        "Classic winding",
        "HotAndCold",
        Viewport(0.02, 0.98, 1.0, 4.4),
        320,
        650,
        0.95,
        note="Broad high-coupling winding bands in a diverging palette.",
    ),
)


def get_reference_preset(label: str) -> ReferencePreset:
    for preset in REFERENCE_PRESETS:
        if preset.label == label:
            return preset
    raise ValueError(f"Unknown reference preset: {label}")
