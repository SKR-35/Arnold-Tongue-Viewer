"""Parameter-plane rendering independent of the Tkinter interface.

The image shows estimated rotation number modulo one. Equal colors do not
prove mode locking; finite iteration counts limit numerical precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import CircleMapSettings, accelerated_rotation_number_grid
from .palettes import colorize


@dataclass(frozen=True, slots=True)
class Viewport:
    """Visible rectangle in (omega, coupling) parameter space."""

    omega_min: float = 0.0
    omega_max: float = 1.0
    coupling_min: float = 0.0
    coupling_max: float = 1.5

    def __post_init__(self) -> None:
        bounds = (self.omega_min, self.omega_max, self.coupling_min, self.coupling_max)
        if not np.isfinite(bounds).all():
            raise ValueError("viewport bounds must be finite")
        if self.omega_min >= self.omega_max or self.coupling_min >= self.coupling_max:
            raise ValueError("viewport bounds must increase")


def parameter_grid(view: Viewport, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D grids with coupling increasing upwards in the image."""
    if width < 2 or height < 2:
        raise ValueError("width and height must be at least 2")
    xs = np.linspace(view.omega_min, view.omega_max, width)
    ys = np.linspace(view.coupling_max, view.coupling_min, height)
    return np.meshgrid(xs, ys)


def rotation_field(
    view: Viewport,
    width: int,
    height: int,
    settings: CircleMapSettings | None = None,
) -> np.ndarray:
    """Calculate finite-time winding estimates for every pixel."""
    omegas, couplings = parameter_grid(view, width, height)
    return accelerated_rotation_number_grid(omegas, couplings, settings)


def render_rgb(
    view: Viewport,
    width: int,
    height: int,
    settings: CircleMapSettings | None = None,
    *,
    palette: str = "Jade",
    gamma: float = 1.0,
) -> np.ndarray:
    """Render a color image (height, width, 3) with no GUI dependencies."""
    field = rotation_field(view, width, height, settings)
    # Modulo one creates a periodic hue convention for the rotation number.
    return colorize(np.mod(field, 1.0), palette=palette, gamma=gamma)
