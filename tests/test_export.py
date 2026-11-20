"""Headless image export tests."""

import numpy as np
import pytest
from PIL import Image

from arnold_tongue_viewer.export import save_image


@pytest.mark.parametrize("suffix,expected", [("png", "PNG"), ("jpg", "JPEG"), ("jpeg", "JPEG")])
def test_save_image(tmp_path, suffix, expected):
    rgb = np.zeros((12, 18, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    path = save_image(rgb, tmp_path / "nested" / f"image.{suffix}")
    with Image.open(path) as image:
        assert image.format == expected
        assert image.size == (18, 12)


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 3), dtype=float)]
)
def test_reject_invalid_array(tmp_path, bad):
    with pytest.raises(ValueError):
        save_image(bad, tmp_path / "bad.png")


def test_reject_invalid_suffix(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "image.gif")


def test_reject_invalid_quality(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "image.jpg", jpeg_quality=101)
