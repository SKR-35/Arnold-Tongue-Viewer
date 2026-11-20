"""Dependency-light RGB palettes for rotation-number visualizations."""

from __future__ import annotations

import numpy as np

PALETTES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "Jade": ((3, 19, 20), (9, 74, 69), (18, 151, 125), (133, 226, 182), (244, 252, 226)),
    "Sunset": ((24, 17, 65), (91, 34, 113), (202, 64, 102), (255, 168, 93), (255, 237, 178)),
    "Ocean": ((5, 18, 48), (15, 71, 119), (24, 149, 173), (130, 219, 207), (237, 250, 246)),
    "Grayscale": ((0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 255, 255)),
}


def colorize(values: np.ndarray, palette: str = "Jade", gamma: float = 1.0) -> np.ndarray:
    """Map a normalized [0, 1] field to an uint8 RGB array."""
    if palette not in PALETTES:
        raise ValueError(f"Unknown palette: {palette}")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("values must be a finite 2D array")
    data = np.clip(data, 0.0, 1.0) ** gamma
    stops = np.asarray(PALETTES[palette], dtype=np.float64)
    position = data * (len(stops) - 1)
    lower = np.minimum(position.astype(np.intp), len(stops) - 2)
    weight = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - weight) + stops[lower + 1] * weight).astype(np.uint8)
