"""Tests for omega sweep and GIF export."""

import numpy as np
import pytest
from PIL import Image

from arnold_tongue_viewer.bifurcation import (
    BifurcationAnimationOptions,
    BifurcationOptions,
    bifurcation_omega_frames,
    save_bifurcation_gif,
)


def test_omega_trajectory():
    opts = BifurcationAnimationOptions(omega_start=0.2, omega_end=0.5, frames=4)
    assert np.allclose(bifurcation_omega_frames(opts), [0.2, 0.3, 0.4, 0.5])


@pytest.mark.parametrize("frames", [0, 1, 121])
def test_invalid_frames(frames):
    with pytest.raises(ValueError, match="Frame count"):
        BifurcationAnimationOptions(frames=frames)


def test_gif_export(tmp_path):
    path = tmp_path / "bifurcation.gif"
    progress = []
    save_bifurcation_gif(
        str(path),
        BifurcationOptions(transient=8, samples=12),
        BifurcationAnimationOptions(frames=3, width=80, height=64),
        on_progress=lambda i, n: progress.append((i, n)),
    )
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3
        assert image.size == (80, 64)
    assert progress == [(1, 3), (2, 3), (3, 3)]
