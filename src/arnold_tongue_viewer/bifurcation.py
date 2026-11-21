"""One-parameter bifurcation diagrams for circle maps.

Horizontal axis: coupling K. Vertical axis: asymptotic phase x mod 1.
For a fixed omega, each coupling column plots post-transient orbit samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .map_families import MAP_FAMILIES, forcing
from .palettes import colorize


@dataclass(frozen=True)
class BifurcationOptions:
    omega: float = 0.35
    k_min: float = 0.0
    k_max: float = 4.0
    transient: int = 400
    samples: int = 240
    theta0: float = 0.123
    phase_min: float = 0.0
    phase_max: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite((self.omega, self.k_min, self.k_max, self.theta0)).all():
            raise ValueError("Bifurcation parameters must be finite")
        if not np.isfinite((self.phase_min, self.phase_max)).all():
            raise ValueError("Phase limits must be finite")
        if not (0 <= self.phase_min < self.phase_max <= 1):
            raise ValueError("Phase range must lie inside [0, 1]")
        if self.k_max <= self.k_min:
            raise ValueError("K maximum must exceed K minimum")
        if not (0 <= self.transient <= 100000 and 1 <= self.samples <= 100000):
            raise ValueError("Invalid iteration counts")


def bifurcation_orbits(
    width: int, options: BifurcationOptions, *, family: str = "Sine"
) -> np.ndarray:
    """Return samples x column of wrapped orbit phases, shape (samples, width)."""
    if family not in MAP_FAMILIES:
        raise ValueError(f"Unknown map family: {family}")
    if width < 2:
        raise ValueError("Width must be at least 2")
    coupling = np.linspace(options.k_min, options.k_max, width)
    factor = coupling / (2 * np.pi)
    phase = np.full(width, options.theta0 % 1.0, dtype=float)
    for _ in range(options.transient):
        phase = (phase + options.omega - factor * forcing(phase, family)) % 1.0
    result = np.empty((options.samples, width), dtype=float)
    for i in range(options.samples):
        phase = (phase + options.omega - factor * forcing(phase, family)) % 1.0
        result[i] = phase
    return result


def bifurcation_density(
    width: int, height: int, options: BifurcationOptions, *, family: str = "Sine"
) -> np.ndarray:
    """Histogram of orbit visits within the visible phase interval."""
    if height < 2:
        raise ValueError("Height must be at least 2")
    orbits = bifurcation_orbits(width, options, family=family)
    visible = (orbits >= options.phase_min) & (orbits <= options.phase_max)
    phase = (orbits - options.phase_min) / (options.phase_max - options.phase_min)
    rows = np.clip((phase * height).astype(np.intp), 0, height - 1)
    cols = np.broadcast_to(np.arange(width), rows.shape)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (height - 1 - rows[visible], cols[visible]), 1)
    return np.clip(np.log1p(counts) / np.log1p(max(2, options.samples)) * 2.8, 0, 1)


def render_bifurcation(
    width: int,
    height: int,
    options: BifurcationOptions,
    *,
    family: str = "Sine",
    palette: str = "Classic",
) -> np.ndarray:
    """Render bifurcation density with a selectable color palette."""
    brightness = bifurcation_density(width, height, options, family=family)
    if palette == "Classic":
        background = np.array([6, 8, 23], dtype=float)
        foreground = np.array([255, 190, 110], dtype=float)
        return np.uint8(np.rint(background + brightness[..., None] * (foreground - background)))
    return colorize(brightness, palette)


def save_bifurcation(
    path: str, width: int, height: int, options: BifurcationOptions, *, family: str = "Sine"
) -> None:
    Image.fromarray(render_bifurcation(width, height, options, family=family)).save(path)
