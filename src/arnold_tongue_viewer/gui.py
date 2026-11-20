"""Tkinter explorer for the Arnold tongue parameter plane."""

from __future__ import annotations

import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .core import HAS_NUMBA, CircleMapSettings
from .palettes import PALETTES
from .render import Viewport, render_rgb


def zoom_view(view: Viewport, factor: float, x_fraction: float, y_fraction: float) -> Viewport:
    """Zoom around a fractional canvas position (y=0 at top)."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("zoom factor must be finite and positive")
    if not (0 <= x_fraction <= 1 and 0 <= y_fraction <= 1):
        raise ValueError("anchor must lie within the canvas")
    x = view.omega_min + x_fraction * (view.omega_max - view.omega_min)
    y = view.coupling_max - y_fraction * (view.coupling_max - view.coupling_min)
    return Viewport(
        x + (view.omega_min - x) * factor,
        x + (view.omega_max - x) * factor,
        y + (view.coupling_min - y) * factor,
        y + (view.coupling_max - y) * factor,
    )


def pan_view(view: Viewport, dx_fraction: float, dy_fraction: float) -> Viewport:
    """Shift view according to screen drag fractions."""
    dx = dx_fraction * (view.omega_max - view.omega_min)
    dy = dy_fraction * (view.coupling_max - view.coupling_min)
    return Viewport(
        view.omega_min - dx,
        view.omega_max - dx,
        view.coupling_min + dy,
        view.coupling_max + dy,
    )


class ArnoldViewer:
    """Interactive parameter-plane viewer; all Tk operations stay on the main thread."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Arnold Tongue Viewer")
        self.root.geometry("1100x750")
        self.view = Viewport()
        self.settings = CircleMapSettings(transient=100, samples=250)
        self.palette = tk.StringVar(value="Jade")
        self.gamma = tk.DoubleVar(value=1.0)
        self.transient = tk.StringVar(value="100")
        self.samples = tk.StringVar(value="250")
        self.status = tk.StringVar(value="Ready")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arnold-render")
        self._results: queue.SimpleQueue = queue.SimpleQueue()
        self._generation = 0
        self._pending: str | None = None
        self._refine_pending: str | None = None
        self._image: ImageTk.PhotoImage | None = None
        self._drag: tuple[int, int] | None = None
        self._closed = False

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Zoom +", command=lambda: self._zoom(0.7)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Zoom -", command=lambda: self._zoom(1 / 0.7)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, text="Palette").pack(side=tk.LEFT)
        ttk.OptionMenu(
            controls, self.palette, "Jade", *PALETTES, command=lambda _: self.request_render()
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Gamma").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Scale(
            controls,
            from_=0.5,
            to=2.5,
            variable=self.gamma,
            command=lambda _: self._schedule_render(),
            length=120,
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Transient").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.transient, width=6).pack(side=tk.LEFT)
        ttk.Label(controls, text="Samples").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.samples, width=6).pack(side=tk.LEFT)
        ttk.Button(controls, text="Apply", command=self._apply_settings).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.root, background="#07151b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.root, textvariable=self.status, padding=5).pack(fill=tk.X)
        self.canvas.bind("<Configure>", lambda _: self._schedule_render())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(0.8, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.25, event.x, event.y))
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-1>", lambda _: self._end_pan())
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(60, self._poll_results)
        self._schedule_render()

    def _dimensions(self) -> tuple[int, int]:
        return max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())

    def _schedule_render(self) -> None:
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        if self._refine_pending is not None:
            self.root.after_cancel(self._refine_pending)
            self._refine_pending = None
        self._pending = self.root.after(180, self.request_render)

    def request_render(self) -> None:
        self._pending = None
        if self._refine_pending is not None:
            self.root.after_cancel(self._refine_pending)
            self._refine_pending = None
        self._generation += 1
        generation = self._generation
        width, height = self._dimensions()
        view, settings = self.view, self.settings
        palette, gamma = self.palette.get(), self.gamma.get()
        self.status.set(
            f"Rendering preview…  Ω [{view.omega_min:.3f}, {view.omega_max:.3f}] "
            f"K [{view.coupling_min:.3f}, {view.coupling_max:.3f}] "
            f"Numba={'ON' if HAS_NUMBA else 'OFF'}"
        )

        def submit_render(quality: str) -> None:
            if generation != self._generation or self._closed:
                return
            if quality == "preview":
                scale = min(1.0, 320 / width, 240 / height)
            else:
                scale = 1.0
            rw, rh = max(2, int(width * scale)), max(2, int(height * scale))

            def work() -> None:
                try:
                    rgb = render_rgb(view, rw, rh, settings, palette=palette, gamma=gamma)
                    image = Image.fromarray(rgb)
                    if (rw, rh) != (width, height):
                        image = image.resize((width, height), Image.Resampling.BILINEAR)
                    self._results.put((generation, quality, image, None))
                except Exception as exc:
                    self._results.put((generation, quality, None, str(exc)))

            self._executor.submit(work)

        submit_render("preview")
        self._refine_pending = self.root.after(400, lambda: submit_render("full"))

    def _poll_results(self) -> None:
        if self._closed:
            return
        while not self._results.empty():
            generation, quality, image, error = self._results.get()
            if generation != self._generation:
                continue
            if error:
                self.status.set(f"Render failed: {error}")
            elif image is not None:
                self._image = ImageTk.PhotoImage(image)
                self.canvas.delete("image")
                self.canvas.create_image(0, 0, image=self._image, anchor="nw", tags="image")
                self.status.set(
                    f"{quality.title()} ready · Ω [{self.view.omega_min:.3f}, "
                    f"{self.view.omega_max:.3f}] · K [{self.view.coupling_min:.3f}, "
                    f"{self.view.coupling_max:.3f}] · Numba={'ON' if HAS_NUMBA else 'OFF'}"
                )
        self.root.after(60, self._poll_results)

    def _apply_settings(self) -> None:
        try:
            transient, samples = int(self.transient.get()), int(self.samples.get())
            if not (0 <= transient <= 100_000 and 1 <= samples <= 100_000):
                raise ValueError("Iteration counts are out of range")
            self.settings = replace(self.settings, transient=transient, samples=samples)
        except ValueError as exc:
            messagebox.showerror("Invalid iterations", str(exc))
            return
        self.request_render()

    def _zoom(self, factor: float) -> None:
        self._zoom_at(factor, *[v / 2 for v in self._dimensions()])

    def _zoom_at(self, factor: float, x: float, y: float) -> None:
        w, h = self._dimensions()
        self.view = zoom_view(self.view, factor, min(1, max(0, x / w)), min(1, max(0, y / h)))
        self._schedule_render()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(0.8 if event.delta > 0 else 1.25, event.x, event.y)

    def _start_pan(self, event: tk.Event) -> None:
        self._drag = (event.x, event.y)

    def _pan(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        x, y = self._drag
        w, h = self._dimensions()
        self.view = pan_view(self.view, (event.x - x) / w, (event.y - y) / h)
        self._drag = (event.x, event.y)
        self._schedule_render()

    def _end_pan(self) -> None:
        self._drag = None

    def _reset(self) -> None:
        self.view = Viewport()
        self.request_render()

    def _close(self) -> None:
        self._closed = True
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        if self._refine_pending is not None:
            self.root.after_cancel(self._refine_pending)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
