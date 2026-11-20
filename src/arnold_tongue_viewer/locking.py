"""Conservative finite-time diagnostics for low-denominator mode locking.

A candidate p/q must have both a nearly periodic q-step orbit and an
attracting q-cycle. This is numerical evidence, not an exact proof.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from .core import HAS_NUMBA, CircleMapSettings, njit, prange
from .render import Viewport, parameter_grid


def rational_labels(max_denominator: int = 8) -> tuple[tuple[int, int], ...]:
    if not 1 <= max_denominator <= 32:
        raise ValueError("max_denominator must be between 1 and 32")
    return tuple(
        (p, q) for q in range(1, max_denominator + 1) for p in range(q + 1) if gcd(p, q) == 1
    )


def _classify_point(
    omega: float,
    coupling: float,
    transient: int,
    samples: int,
    max_denominator: int,
    tolerance: float,
) -> tuple[int, int]:
    """Return (numerator, denominator), or (-1, 0) when not confidently locked."""
    tau = 2.0 * np.pi
    phase = 0.0
    factor = coupling / tau
    for _ in range(transient + samples):
        phase = (phase + omega - factor * np.sin(tau * phase)) % 1.0
    for q in range(1, max_denominator + 1):
        x = phase
        winding = 0.0
        multiplier = 1.0
        for _ in range(q):
            delta = omega - factor * np.sin(tau * x)
            winding += delta
            multiplier *= 1.0 - coupling * np.cos(tau * x)
            x = (x + delta) % 1.0
        p = int(round(winding))
        if (
            0 <= p <= q
            and gcd(p, q) == 1
            and abs(winding - p) <= tolerance * q
            and abs(multiplier) < 0.95
        ):
            return p, q
    return -1, 0


if HAS_NUMBA:
    _classify_point_jit = njit(cache=True)(_classify_point)

    @njit(cache=True, parallel=True)
    def _locking_grid_jit(omega, coupling, transient, samples, denominator, tolerance):
        labels = np.zeros(omega.shape, dtype=np.int16)
        for i in prange(omega.shape[0]):
            for j in range(omega.shape[1]):
                p, q = _classify_point_jit(
                    omega[i, j], coupling[i, j], transient, samples, denominator, tolerance
                )
                if q:
                    labels[i, j] = q * 100 + p + 1
        return labels


def locking_field(
    view: Viewport,
    width: int,
    height: int,
    settings: CircleMapSettings | None = None,
    *,
    max_denominator: int = 8,
    tolerance: float = 0.002,
) -> np.ndarray:
    """Zero means unclassified; positive labels encode q*100+p+1."""
    rational_labels(max_denominator)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 0.05:
        raise ValueError("tolerance must be in (0, 0.05]")
    config = settings or CircleMapSettings()
    omega, coupling = parameter_grid(view, width, height)
    if HAS_NUMBA:
        return _locking_grid_jit(
            np.ascontiguousarray(omega),
            np.ascontiguousarray(coupling),
            config.transient,
            config.samples,
            max_denominator,
            tolerance,
        )
    labels = np.zeros(omega.shape, dtype=np.int16)
    for i in range(height):
        for j in range(width):
            p, q = _classify_point(
                float(omega[i, j]),
                float(coupling[i, j]),
                config.transient,
                config.samples,
                max_denominator,
                tolerance,
            )
            if q:
                labels[i, j] = q * 100 + p + 1
    return labels


def colorize_locking(labels: np.ndarray, palette: str = "Yikique") -> np.ndarray:
    """Assign distinct colors to rational plateaus, dark background otherwise."""
    from .palettes import PALETTES

    if palette not in PALETTES:
        raise ValueError(f"Unknown palette: {palette}")
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a 2D integer array")
    stops = np.asarray(PALETTES[palette], dtype=np.uint8)
    result = np.empty((*values.shape, 3), dtype=np.uint8)
    result[:] = (10, 12, 25)
    for code in np.unique(values):
        if code <= 0:
            continue
        q = int((code - 1) // 100)
        p = int((code - 1) % 100)
        # Deterministic hash of the rational pair; stable across zoom and pan.
        index = (p * 19 + q * 7) % len(stops)
        result[values == code] = stops[index]
    return result
