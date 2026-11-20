"""Save rendered RGB frames without depending on the Tkinter GUI."""

from __future__ import annotations

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
