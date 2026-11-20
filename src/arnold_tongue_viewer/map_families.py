"""Alternative circle-map forcing families, independent of GUI and legacy core.

All families use the same normalization:
    F(x) = x + omega - (K / (2*pi)) * forcing(x mod 1).
The forcing functions are periodic, odd, and have unit peak magnitude.
This keeps K comparable across models, but does not reproduce every
normalization convention used in external reference images.
"""

from __future__ import annotations

import numpy as np

from .core import HAS_NUMBA, CircleMapSettings, njit, prange

MAP_FAMILIES = ("Sine", "Triangle", "Sawtooth")
_TAU = 2.0 * np.pi


def _family_id(family: str) -> int:
    try:
        return MAP_FAMILIES.index(family)
    except ValueError as exc:
        raise ValueError(f"Unknown map family: {family}") from exc


def forcing(phase: np.ndarray | float, family: str = "Sine") -> np.ndarray | float:
    """Unit-amplitude periodic forcing; sawtooth has a jump at phase 0.5."""
    kind = _family_id(family)
    x = np.asarray(phase, dtype=float) % 1.0
    if kind == 0:
        result = np.sin(_TAU * x)
    elif kind == 1:
        result = np.where(x < 0.25, 4.0 * x, np.where(x < 0.75, 2.0 - 4.0 * x, 4.0 * x - 4.0))
    else:
        result = np.where(x < 0.5, 2.0 * x, 2.0 * x - 2.0)
    return float(result) if np.ndim(phase) == 0 else result


def map_step(phase: float, omega: float, coupling: float, family: str = "Sine") -> float:
    """Return the unwrapped lift after one iteration."""
    if not np.isfinite((phase, omega, coupling)).all():
        raise ValueError("Parameters must be finite")
    return float(phase + omega - coupling / _TAU * forcing(phase, family))


def _forcing_scalar(phase: float, kind: int) -> float:
    x = phase % 1.0
    if kind == 0:
        return np.sin(_TAU * x)
    if kind == 1:
        if x < 0.25:
            return 4.0 * x
        if x < 0.75:
            return 2.0 - 4.0 * x
        return 4.0 * x - 4.0
    return 2.0 * x if x < 0.5 else 2.0 * x - 2.0


def _rotation_scalar(
    omega: float, coupling: float, transient: int, samples: int, theta0: float, kind: int
) -> float:
    phase = theta0 % 1.0
    factor = coupling / _TAU
    for _ in range(transient):
        phase = (phase + omega - factor * _forcing_scalar(phase, kind)) % 1.0
    displacement = 0.0
    for _ in range(samples):
        delta = omega - factor * _forcing_scalar(phase, kind)
        displacement += delta
        phase = (phase + delta) % 1.0
    return displacement / samples


if HAS_NUMBA:
    _forcing_scalar = njit(cache=True)(_forcing_scalar)
    _rotation_scalar_jit = njit(cache=True)(_rotation_scalar)

    @njit(cache=True, parallel=True)
    def _family_grid_jit(omega, coupling, transient, samples, theta0, kind):
        result = np.empty(omega.shape, dtype=np.float64)
        flat_o, flat_k, flat_r = omega.ravel(), coupling.ravel(), result.ravel()
        for i in prange(flat_o.size):
            flat_r[i] = _rotation_scalar_jit(flat_o[i], flat_k[i], transient, samples, theta0, kind)
        return result


def family_rotation_grid(
    omegas: np.ndarray,
    couplings: np.ndarray,
    settings: CircleMapSettings | None = None,
    *,
    family: str = "Sine",
) -> np.ndarray:
    """Compute winding estimates on broadcast-compatible parameter arrays."""
    kind = _family_id(family)
    config = settings or CircleMapSettings()
    omega, coupling = np.broadcast_arrays(
        np.asarray(omegas, dtype=float), np.asarray(couplings, dtype=float)
    )
    if not np.isfinite(omega).all() or not np.isfinite(coupling).all():
        raise ValueError("Parameter arrays must be finite")
    omega = np.ascontiguousarray(omega)
    coupling = np.ascontiguousarray(coupling)
    if HAS_NUMBA:
        return _family_grid_jit(
            omega, coupling, config.transient, config.samples, config.theta0, kind
        )
    result = np.empty(omega.shape, dtype=float)
    for idx in np.ndindex(omega.shape):
        result[idx] = _rotation_scalar(
            float(omega[idx]),
            float(coupling[idx]),
            config.transient,
            config.samples,
            config.theta0,
            kind,
        )
    return result
