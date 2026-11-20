"""Viewport-based GIF animation, independent of the Tkinter interface."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .core import CircleMapSettings
from .locking import colorize_locking, locking_field
from .render import Viewport, render_rgb


@dataclass(frozen=True)
class AnimationOptions:
    frames: int = 24
    width: int = 640
    height: int = 420
    duration_ms: int = 100
    zoom_factor: float = 0.3
    motion: str = "Smooth Zoom"
    color_effect: str = "Static"
    second_palette: str = "Jade"
    seed: int = 42

    def __post_init__(self) -> None:
        if not 2 <= self.frames <= 120:
            raise ValueError("frames must be between 2 and 120")
        if not 64 <= self.width <= 1920 or not 64 <= self.height <= 1080:
            raise ValueError("invalid animation dimensions")
        if not 20 <= self.duration_ms <= 2000:
            raise ValueError("invalid frame duration")
        if self.motion not in {"Smooth Zoom", "Orbit", "Pendulum", "Random Walk"}:
            raise ValueError("unknown animation motion")
        if self.color_effect not in {"Static", "Palette Pulse", "Negative Pulse"}:
            raise ValueError("unknown color effect")
        if not 0 < self.zoom_factor < 1:
            raise ValueError("zoom factor must be between 0 and 1")


def animation_viewports(view: Viewport, options: AnimationOptions) -> tuple[Viewport, ...]:
    """Generate a geometric zoom towards the current viewport center."""
    center_o = (view.omega_min + view.omega_max) / 2
    center_k = (view.coupling_min + view.coupling_max) / 2
    half_o = (view.omega_max - view.omega_min) / 2
    half_k = (view.coupling_max - view.coupling_min) / 2
    result = []
    rng = np.random.default_rng(options.seed)
    offsets = rng.normal(size=(options.frames, 2))
    offsets = np.cumsum(offsets, axis=0)
    offsets -= offsets[0]
    peak = max(float(np.max(np.abs(offsets))), 1.0)
    offsets /= peak
    for index, t in enumerate(np.linspace(0.0, 1.0, options.frames)):
        phase = 2 * math.pi * float(t)
        scale = options.zoom_factor ** float(t)
        # Motion is measured relative to the original viewport dimensions.
        # Fade offsets to zero at both ends for a clean beginning/end.
        envelope = math.sin(math.pi * float(t))
        if options.motion == "Orbit":
            dx = 0.18 * envelope * (math.cos(phase) - 1)
            dy = 0.18 * envelope * math.sin(phase)
        elif options.motion == "Pendulum":
            dx, dy = 0.23 * envelope * math.sin(2 * phase), 0.0
        elif options.motion == "Random Walk":
            dx, dy = 0.18 * envelope * offsets[index]
        else:
            dx, dy = 0.0, 0.0
        x = center_o + dx * half_o * 2
        y = center_k + dy * half_k * 2
        result.append(
            Viewport(
                x - half_o * scale,
                x + half_o * scale,
                y - half_k * scale,
                y + half_k * scale,
            )
        )
    return tuple(result)


def render_animation_frame(
    view: Viewport,
    settings: CircleMapSettings,
    *,
    width: int,
    height: int,
    palette: str,
    mode: str,
    gamma: float,
) -> Image.Image:
    if mode == "Mode locking":
        rgb = colorize_locking(locking_field(view, width, height, settings), palette)
    elif mode == "Rotation number":
        rgb = render_rgb(view, width, height, settings, palette=palette, gamma=gamma)
    else:
        raise ValueError(f"Unknown display mode: {mode}")
    return Image.fromarray(rgb)


def save_zoom_gif(
    destination: str | Path,
    view: Viewport,
    settings: CircleMapSettings,
    *,
    palette: str,
    mode: str,
    gamma: float,
    options: AnimationOptions = AnimationOptions(),
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Render genuine parameter-space zoom frames and write an animated GIF."""
    path = Path(destination)
    if path.suffix.lower() != ".gif":
        raise ValueError("Animation export requires a .gif filename")
    images = []
    for index, frame_view in enumerate(animation_viewports(view, options), start=1):
        frame = render_animation_frame(
            frame_view,
            settings,
            width=options.width,
            height=options.height,
            palette=palette,
            mode=mode,
            gamma=gamma,
        )
        if options.color_effect != "Static":
            # Sinusoidal brightness mixing avoids sudden color jumps.
            weight = (1 - math.cos(2 * math.pi * (index - 1) / options.frames)) / 2
            if options.color_effect == "Palette Pulse":
                other = render_animation_frame(
                    frame_view,
                    settings,
                    width=options.width,
                    height=options.height,
                    palette=options.second_palette,
                    mode=mode,
                    gamma=gamma,
                )
                frame = Image.blend(frame.convert("RGB"), other.convert("RGB"), weight)
            else:
                from PIL import ImageOps

                frame = Image.blend(
                    frame.convert("RGB"), ImageOps.invert(frame.convert("RGB")), weight
                )
        images.append(frame)
        if on_progress:
            on_progress(index, options.frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=options.duration_ms,
        loop=0,
        optimize=False,
    )
    return path
