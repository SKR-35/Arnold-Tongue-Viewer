from pathlib import Path

import pytest
from PIL import Image

from arnold_tongue_viewer.animation import (
    AnimationOptions,
    animation_viewports,
    save_zoom_gif,
)
from arnold_tongue_viewer.core import CircleMapSettings
from arnold_tongue_viewer.render import Viewport


def test_zoom_endpoints_and_monotonicity():
    view = Viewport()
    options = AnimationOptions(frames=6, zoom_factor=0.25)
    frames = animation_viewports(view, options)
    assert frames[0] == view
    widths = [f.omega_max - f.omega_min for f in frames]
    assert all(a > b for a, b in zip(widths, widths[1:], strict=False))
    assert widths[-1] == pytest.approx(widths[0] * 0.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frames": 1},
        {"frames": 121},
        {"zoom_factor": 0},
        {"zoom_factor": 1},
        {"width": 12},
        {"duration_ms": 0},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        AnimationOptions(**kwargs)


def test_gif_export(tmp_path: Path, monkeypatch):
    from arnold_tongue_viewer import animation

    def fake_frame(*args, width, height, **kwargs):
        return Image.new("RGB", (width, height), (int(args[0].omega_min * 200) % 256, 50, 80))

    monkeypatch.setattr(animation, "render_animation_frame", fake_frame)
    path = tmp_path / "zoom.gif"
    progress = []
    save_zoom_gif(
        path,
        Viewport(),
        CircleMapSettings(),
        palette="Yikique",
        mode="Mode locking",
        gamma=1.0,
        options=AnimationOptions(frames=4, width=64, height=64),
        on_progress=lambda n, total: progress.append((n, total)),
    )
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 4
        assert gif.info["loop"] == 0
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


@pytest.mark.parametrize("motion", ["Smooth Zoom", "Orbit", "Pendulum", "Random Walk"])
def test_motion_paths(motion):
    opts = AnimationOptions(frames=12, motion=motion, seed=7)
    points = animation_viewports(Viewport(), opts)
    assert len(points) == 12
    assert points == animation_viewports(Viewport(), opts)
    if motion != "Smooth Zoom":
        assert any(abs((p.omega_min + p.omega_max) / 2 - 0.5) > 1e-5 for p in points[1:-1])


def test_random_seed_changes_path():
    a = animation_viewports(Viewport(), AnimationOptions(motion="Random Walk", seed=1))
    b = animation_viewports(Viewport(), AnimationOptions(motion="Random Walk", seed=2))
    assert a != b


@pytest.mark.parametrize("effect", ["Static", "Palette Pulse", "Negative Pulse"])
def test_color_effect_gif(tmp_path, monkeypatch, effect):
    from arnold_tongue_viewer import animation

    def frame(view, settings, *, width, height, palette, mode, gamma):
        return Image.new(
            "RGB", (width, height), (80, 110, 150) if palette == "Yikique" else (220, 50, 20)
        )

    monkeypatch.setattr(animation, "render_animation_frame", frame)
    path = tmp_path / f"{effect}.gif"
    save_zoom_gif(
        path,
        Viewport(),
        CircleMapSettings(),
        palette="Yikique",
        mode="Mode locking",
        gamma=1.0,
        options=AnimationOptions(frames=8, width=64, height=64, color_effect=effect),
    )
    with Image.open(path) as gif:
        assert gif.n_frames == (1 if effect == "Static" else 8)
