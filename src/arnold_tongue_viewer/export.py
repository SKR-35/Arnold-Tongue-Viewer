"""Save rendered RGB frames without depending on the Tkinter GUI."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


def save_image(rgb: np.ndarray, destination: str | Path, *, jpeg_quality: int = 95) -> Path:
    """Save an RGB uint8 image as PNG or JPEG, creating parent directories."""
    path = Path(destination)
    image_format = SUPPORTED_FORMATS.get(path.suffix.lower())
    if image_format is None:
        raise ValueError("Export format must be PNG, JPG, or JPEG")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB array with shape (height, width, 3)")
    if rgb.dtype != np.uint8 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("Expected a non-empty uint8 RGB array")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"quality": jpeg_quality, "subsampling": 0} if image_format == "JPEG" else {}
    Image.fromarray(rgb, mode="RGB").save(path, format=image_format, **options)
    return path


def suggested_filename(
    *,
    mode: str,
    palette: str,
    gamma: float,
    transient: int,
    samples: int,
    omega_min: float,
    omega_max: float,
    coupling_min: float,
    coupling_max: float,
    extension: str = ".png",
) -> str:
    """Return a portable, reproducible export name based on rendering settings."""
    if extension.lower() not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported image extension")
    values = (gamma, omega_min, omega_max, coupling_min, coupling_max)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Filename parameters must be finite")
    if transient < 0 or samples < 1 or omega_min >= omega_max or coupling_min >= coupling_max:
        raise ValueError("Invalid rendering parameters")

    def slug(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        if not cleaned:
            raise ValueError("Mode and palette must have a name")
        return cleaned

    def number(value: float) -> str:
        return f"{value:.4f}"

    mode_slug = slug(mode)
    fields = ["arnold", mode_slug, slug(palette)]
    if mode_slug != "mode-locking":
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        fields.append(f"g{gamma:.2f}")
    fields += [
        f"t{transient}",
        f"s{samples}",
        f"O{number(omega_min)}-{number(omega_max)}",
        f"K{number(coupling_min)}-{number(coupling_max)}",
    ]
    return "_".join(fields) + extension.lower()
