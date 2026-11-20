"""Numerical core for the standard sine circle map.

The lift (without modulo) is used to measure net winding. The wrapped phase
is used only when evaluating the periodic sine term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CircleMapSettings:
    """Parameters shared by scalar and parameter-grid calculations."""

    transient: int = 300
    samples: int = 1000
    theta0: float = 0.0

    def __post_init__(self) -> None:
        if self.transient < 0:
            raise ValueError("transient must be non-negative")
        if self.samples < 1:
            raise ValueError("samples must be positive")
        if not np.isfinite(self.theta0):
            raise ValueError("theta0 must be finite")


def circle_map_step(theta: float, omega: float, coupling: float) -> float:
    """Return one *unwrapped* step of the sine circle-map lift.

    F(theta) = theta + omega - K/(2*pi) * sin(2*pi*theta).
    """
    if not np.isfinite([theta, omega, coupling]).all():
        raise ValueError("theta, omega and coupling must be finite")
    return float(theta + omega - coupling / (2.0 * np.pi) * np.sin(2.0 * np.pi * theta))


def rotation_number(
    omega: float,
    coupling: float,
    settings: CircleMapSettings | None = None,
) -> float:
    """Estimate winding per iteration after discarding a transient.

    This is a finite-time approximation, not an exact rationality classifier.
    """
    if not np.isfinite([omega, coupling]).all():
        raise ValueError("omega and coupling must be finite")
    config = settings or CircleMapSettings()
    phase = config.theta0 % 1.0
    for _ in range(config.transient):
        phase = (phase + omega - coupling / (2.0 * np.pi) * np.sin(2.0 * np.pi * phase)) % 1.0

    displacement = 0.0
    for _ in range(config.samples):
        delta = omega - coupling / (2.0 * np.pi) * np.sin(2.0 * np.pi * phase)
        displacement += delta
        phase = (phase + delta) % 1.0
    return float(displacement / config.samples)


def rotation_number_grid(
    omegas: np.ndarray,
    couplings: np.ndarray,
    settings: CircleMapSettings | None = None,
) -> np.ndarray:
    """Compute rotation numbers for broadcast-compatible parameter arrays.

    Arrays can be created using np.meshgrid; no GUI or color mapping is involved.
    """
    config = settings or CircleMapSettings()
    omega, coupling = np.broadcast_arrays(
        np.asarray(omegas, dtype=np.float64),
        np.asarray(couplings, dtype=np.float64),
    )
    if not np.isfinite(omega).all() or not np.isfinite(coupling).all():
        raise ValueError("parameter arrays must contain only finite values")

    phase = np.full(omega.shape, config.theta0 % 1.0, dtype=np.float64)
    factor = coupling / (2.0 * np.pi)
    for _ in range(config.transient):
        phase = (phase + omega - factor * np.sin(2.0 * np.pi * phase)) % 1.0

    displacement = np.zeros(omega.shape, dtype=np.float64)
    for _ in range(config.samples):
        delta = omega - factor * np.sin(2.0 * np.pi * phase)
        displacement += delta
        phase = (phase + delta) % 1.0
    return displacement / config.samples


# A separate optional JIT kernel avoids changing the established NumPy API.
try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised in installations without numba
    njit = None
    prange = range

HAS_NUMBA = njit is not None

if HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _rotation_grid_jit(omega, coupling, transient, samples, theta0):
        output = np.empty(omega.shape, dtype=np.float64)
        flat_o = omega.ravel()
        flat_k = coupling.ravel()
        flat_result = output.ravel()
        for index in prange(flat_o.size):
            o = flat_o[index]
            factor = flat_k[index] / (2.0 * np.pi)
            phase = theta0 % 1.0
            for _ in range(transient):
                phase = (phase + o - factor * np.sin(2.0 * np.pi * phase)) % 1.0
            displacement = 0.0
            for _ in range(samples):
                delta = o - factor * np.sin(2.0 * np.pi * phase)
                displacement += delta
                phase = (phase + delta) % 1.0
            flat_result[index] = displacement / samples
        return output


def accelerated_rotation_number_grid(omegas, couplings, settings=None):
    """Use Numba when available; otherwise preserve the NumPy implementation."""
    config = settings or CircleMapSettings()
    omega, coupling = np.broadcast_arrays(
        np.asarray(omegas, dtype=np.float64), np.asarray(couplings, dtype=np.float64)
    )
    if not np.isfinite(omega).all() or not np.isfinite(coupling).all():
        raise ValueError("parameter arrays must contain only finite values")
    if not HAS_NUMBA:
        return rotation_number_grid(omega, coupling, config)
    return _rotation_grid_jit(
        np.ascontiguousarray(omega),
        np.ascontiguousarray(coupling),
        config.transient,
        config.samples,
        config.theta0,
    )
