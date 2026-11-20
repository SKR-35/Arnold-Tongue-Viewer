"""Additional parameter-plane observables for the circle-map explorer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import HAS_NUMBA, CircleMapSettings, njit, prange
from .map_families import MAP_FAMILIES, _forcing_scalar
from .palettes import colorize
from .render import Viewport, parameter_grid, rotation_field

EXPLORER_MODES = ("Classic winding", "Poincare recurrence")


@dataclass(frozen=True)
class RecurrenceOptions:
    tolerance: float = 0.015
    max_steps: int = 250

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or not 0 < self.tolerance < 0.5:
            raise ValueError("tolerance must be between 0 and 0.5")
        if not 2 <= self.max_steps <= 10000:
            raise ValueError("max_steps must be between 2 and 10000")


def _return_time(omega, coupling, transient, theta0, tolerance, steps, kind):
    phase = theta0 % 1.0
    factor = coupling / (2.0 * np.pi)
    for _ in range(transient):
        phase = (phase + omega - factor * _forcing_scalar(phase, kind)) % 1.0
    reference = phase
    for step in range(1, steps + 1):
        phase = (phase + omega - factor * _forcing_scalar(phase, kind)) % 1.0
        distance = abs(phase - reference)
        if min(distance, 1.0 - distance) <= tolerance:
            return step
    return 0


if HAS_NUMBA:
    _return_time_jit = njit(cache=True)(_return_time)

    @njit(cache=True, parallel=True)
    def _recurrence_grid_jit(omegas, couplings, transient, theta0, tolerance, steps, kind):
        out = np.empty(omegas.shape, dtype=np.int32)
        flat_o, flat_k, flat_out = omegas.ravel(), couplings.ravel(), out.ravel()
        for i in prange(flat_o.size):
            flat_out[i] = _return_time_jit(
                flat_o[i], flat_k[i], transient, theta0, tolerance, steps, kind
            )
        return out


def recurrence_field(
    view: Viewport,
    width: int,
    height: int,
    settings: CircleMapSettings | None = None,
    *,
    family: str = "Sine",
    options: RecurrenceOptions = RecurrenceOptions(),
) -> np.ndarray:
    """First return to a tolerance-neighborhood of the post-transient phase.

    Zero means no return detected within max_steps, NOT a proof of nonrecurrence.
    """
    if family not in MAP_FAMILIES:
        raise ValueError(f"Unknown map family: {family}")
    config = settings or CircleMapSettings()
    omega, coupling = parameter_grid(view, width, height)
    omega, coupling = np.ascontiguousarray(omega), np.ascontiguousarray(coupling)
    kind = MAP_FAMILIES.index(family)
    if HAS_NUMBA:
        return _recurrence_grid_jit(
            omega,
            coupling,
            config.transient,
            config.theta0,
            options.tolerance,
            options.max_steps,
            kind,
        )
    out = np.empty(omega.shape, dtype=np.int32)
    for idx in np.ndindex(omega.shape):
        out[idx] = _return_time(
            float(omega[idx]),
            float(coupling[idx]),
            config.transient,
            config.theta0,
            options.tolerance,
            options.max_steps,
            kind,
        )
    return out


def render_explorer_rgb(
    view: Viewport,
    width: int,
    height: int,
    settings: CircleMapSettings | None = None,
    *,
    palette: str = "Yikique",
    gamma: float = 1.0,
    family: str = "Sine",
    mode: str = "Classic winding",
    recurrence: RecurrenceOptions = RecurrenceOptions(),
) -> np.ndarray:
    """Render cyclic winding bands or first-return-time coloring."""
    if mode == "Classic winding":
        field = rotation_field(view, width, height, settings, family=family)
        # Frequency multiplication reveals narrow winding plateaus without
        # claiming to classify exact mode locking.
        bands = np.mod(field * 12.0, 1.0)
        return colorize(bands, palette=palette, gamma=gamma)
    if mode == "Poincare recurrence":
        times = recurrence_field(view, width, height, settings, family=family, options=recurrence)
        values = np.log1p(times.astype(float)) / np.log1p(recurrence.max_steps)
        rgb = colorize(values, palette=palette, gamma=gamma)
        rgb[times == 0] = (5, 5, 15)
        return rgb
    raise ValueError(f"Unknown explorer mode: {mode}")
