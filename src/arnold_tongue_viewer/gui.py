"""Tkinter explorer for the Arnold tongue parameter plane."""

from __future__ import annotations

import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .animation import AnimationOptions, save_zoom_gif
from .core import HAS_NUMBA, CircleMapSettings
from .export import save_image, suggested_filename
from .locking import colorize_locking, locking_field
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


def rectangle_view(view: Viewport, x0: float, y0: float, x1: float, y1: float) -> Viewport:
    """Convert normalized screen rectangle to a parameter-space viewport."""
    values = (x0, y0, x1, y1)
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("rectangle coordinates must be finite and within [0, 1]")
    if x0 == x1 or y0 == y1:
        raise ValueError("rectangle must have positive area")
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    dw = view.omega_max - view.omega_min
    dh = view.coupling_max - view.coupling_min
    return Viewport(
        view.omega_min + left * dw,
        view.omega_min + right * dw,
        view.coupling_max - bottom * dh,
        view.coupling_max - top * dh,
    )


class ArnoldViewer:
    """Interactive parameter-plane viewer; all Tk operations stay on the main thread."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Arnold Tongue Viewer")
        self.root.geometry("1100x750")
        self.view = Viewport()
        self.settings = CircleMapSettings(transient=100, samples=250)
        self.palette = tk.StringVar(value="Yikique")
        self.display_mode = tk.StringVar(value="Mode locking")
        self.gamma = tk.DoubleVar(value=1.0)
        self.navigation_mode = tk.StringVar(value="Rectangle zoom")
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
        self._selection_id: int | None = None
        self._closed = False

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Zoom +", command=lambda: self._zoom(0.7)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Zoom -", command=lambda: self._zoom(1 / 0.7)).pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, text="Mouse").pack(side=tk.LEFT)
        ttk.OptionMenu(
            controls,
            self.navigation_mode,
            "Rectangle zoom",
            "Rectangle zoom",
            "Pan",
            command=lambda _: self._cancel_drag(),
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Palette").pack(side=tk.LEFT)
        ttk.OptionMenu(
            controls, self.palette, "Yikique", *PALETTES, command=lambda _: self.request_render()
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Mode").pack(side=tk.LEFT, padx=(10, 2))
        ttk.OptionMenu(
            controls,
            self.display_mode,
            "Mode locking",
            "Mode locking",
            "Rotation number",
            command=lambda _: self._change_display_mode(),
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Gamma").pack(side=tk.LEFT, padx=(10, 2))
        self.gamma_scale = ttk.Scale(
            controls,
            from_=0.5,
            to=2.5,
            variable=self.gamma,
            command=lambda _: self._schedule_render(),
            length=120,
        )
        self.gamma_scale.pack(side=tk.LEFT)
        self.gamma_scale.state(["disabled"])
        ttk.Label(controls, text="Transient").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.transient, width=6).pack(side=tk.LEFT)
        ttk.Label(controls, text="Samples").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.samples, width=6).pack(side=tk.LEFT)
        ttk.Button(controls, text="Apply", command=self._apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Export image", command=self._export_image).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(controls, text="Export GIF", command=self._export_gif).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.root, background="#07151b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(self.root, textvariable=self.status, padding=5).pack(fill=tk.X)
        self.canvas.bind("<Configure>", lambda _: self._schedule_render())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(0.8, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.25, event.x, event.y))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Escape>", lambda _: self._cancel_drag())
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(60, self._poll_results)
        self._schedule_render()

    def _change_display_mode(self) -> None:
        self.gamma_scale.state(
            ["disabled"] if self.display_mode.get() == "Mode locking" else ["!disabled"]
        )
        self.request_render()

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
        display_mode = self.display_mode.get()
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
                    if display_mode == "Mode locking":
                        labels = locking_field(view, rw, rh, settings)
                        rgb = colorize_locking(labels, palette)
                    else:
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
            if generation is None:
                if quality == "animation-progress":
                    self.status.set(f"Rendering GIF frame {error}")
                elif quality == "export":
                    self.status.set(f"Image saved: {error}")
                else:
                    self.status.set(f"Export failed: {error}")
                continue
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

    def _export_image(self) -> None:
        """Export the current view using the actual viewport and display mode."""
        view, settings = self.view, self.settings
        palette, gamma = self.palette.get(), self.gamma.get()
        display_mode = self.display_mode.get()
        filename = suggested_filename(
            mode=display_mode,
            palette=palette,
            gamma=gamma,
            transient=settings.transient,
            samples=settings.samples,
            omega_min=view.omega_min,
            omega_max=view.omega_max,
            coupling_min=view.coupling_min,
            coupling_max=view.coupling_max,
        )
        path = filedialog.asksaveasfilename(
            title="Export Arnold tongue image",
            initialfile=filename,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("JPEG image", "*.jpeg")],
        )
        if not path:
            return
        width, height = self._dimensions()
        view, settings = self.view, self.settings
        palette, gamma = self.palette.get(), self.gamma.get()
        display_mode = self.display_mode.get()
        self.status.set("Exporting image…")

        def work() -> None:
            try:
                if display_mode == "Mode locking":
                    rgb = colorize_locking(locking_field(view, width, height, settings), palette)
                else:
                    rgb = render_rgb(view, width, height, settings, palette=palette, gamma=gamma)
                save_image(rgb, path)
                self._results.put((None, "export", None, str(path)))
            except Exception as exc:
                self._results.put((None, "export-error", None, str(exc)))

        self._executor.submit(work)

    def _export_gif(self) -> None:
        """Export a geometric zoom into the center of the current viewport."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Animation options")
        dialog.transient(self.root)
        dialog.grab_set()
        motion = tk.StringVar(value="Smooth Zoom")
        effect = tk.StringVar(value="Static")
        second = tk.StringVar(value="Jade")
        frames_var = tk.StringVar(value="24")
        zoom_var = tk.StringVar(value="0.30")
        seed_var = tk.StringVar(value="42")
        fields = [
            ("Motion", motion, ["Smooth Zoom", "Orbit", "Pendulum", "Random Walk"]),
            ("Color effect", effect, ["Static", "Palette Pulse", "Negative Pulse"]),
            ("Second palette", second, list(PALETTES)),
        ]
        for row, (label, var, choices) in enumerate(fields):
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=12, pady=5, sticky="w")
            ttk.Combobox(dialog, textvariable=var, values=choices, state="readonly", width=20).grid(
                row=row, column=1, padx=12, pady=5
            )
        for row, (label, var) in enumerate(
            [
                ("Frames (2–120)", frames_var),
                ("Final zoom (0–1)", zoom_var),
                ("Random seed", seed_var),
            ],
            start=3,
        ):
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=12, pady=5, sticky="w")
            ttk.Entry(dialog, textvariable=var, width=23).grid(row=row, column=1, padx=12, pady=5)
        selection = []

        def accept() -> None:
            try:
                options = AnimationOptions(
                    frames=int(frames_var.get()),
                    zoom_factor=float(zoom_var.get()),
                    motion=motion.get(),
                    color_effect=effect.get(),
                    second_palette=second.get(),
                    seed=int(seed_var.get()),
                )
            except ValueError as exc:
                messagebox.showerror("Invalid animation settings", str(exc), parent=dialog)
                return
            selection.append(options)
            dialog.destroy()

        ttk.Button(dialog, text="Export…", command=accept).grid(row=6, column=1, pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=6, column=0, pady=10)
        dialog.wait_window()
        if not selection:
            return
        options = selection[0]
        view, settings = self.view, self.settings
        palette, gamma, mode = self.palette.get(), self.gamma.get(), self.display_mode.get()
        filename = suggested_filename(
            mode=mode,
            palette=palette,
            gamma=gamma,
            transient=settings.transient,
            samples=settings.samples,
            omega_min=view.omega_min,
            omega_max=view.omega_max,
            coupling_min=view.coupling_min,
            coupling_max=view.coupling_max,
        ).removesuffix(".png") + (
            f"_{options.motion.lower().replace(' ', '-')}_z{options.zoom_factor:.2f}"
            f"_{options.color_effect.lower().replace(' ', '-')}_"
            f"{options.second_palette.lower() if options.color_effect == 'Palette Pulse' else 'base'}"
            f"_seed{options.seed if options.motion == 'Random Walk' else 0}"
            f"_{options.frames}f.gif"
        )
        path = filedialog.asksaveasfilename(
            title="Export zoom animation",
            initialfile=filename,
            defaultextension=".gif",
            filetypes=[("Animated GIF", "*.gif")],
        )
        if not path:
            return
        self.status.set("Exporting zoom GIF…")

        def work() -> None:
            try:
                save_zoom_gif(
                    path,
                    view,
                    settings,
                    palette=palette,
                    mode=mode,
                    gamma=gamma,
                    options=options,
                    on_progress=lambda n, total: self._results.put(
                        (None, "animation-progress", None, f"{n}/{total}")
                    ),
                )
                self._results.put((None, "export", None, str(path)))
            except Exception as exc:
                self._results.put((None, "export-error", None, str(exc)))

        self._executor.submit(work)

    def _zoom(self, factor: float) -> None:
        self._zoom_at(factor, *[v / 2 for v in self._dimensions()])

    def _zoom_at(self, factor: float, x: float, y: float) -> None:
        w, h = self._dimensions()
        self.view = zoom_view(self.view, factor, min(1, max(0, x / w)), min(1, max(0, y / h)))
        self._schedule_render()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom_at(0.8 if event.delta > 0 else 1.25, event.x, event.y)

    def _start_drag(self, event: tk.Event) -> None:
        self._cancel_drag()
        self._drag = (event.x, event.y)
        if self.navigation_mode.get() == "Rectangle zoom":
            self._selection_id = self.canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="#ffffff",
                width=2,
                dash=(5, 3),
                tags="selection",
            )

    def _drag_motion(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        x, y = self._drag
        w, h = self._dimensions()
        if self.navigation_mode.get() == "Pan":
            self.view = pan_view(self.view, (event.x - x) / w, (event.y - y) / h)
            self._drag = (event.x, event.y)
            self._schedule_render()
        elif self._selection_id is not None:
            self.canvas.coords(
                self._selection_id,
                x,
                y,
                min(w, max(0, event.x)),
                min(h, max(0, event.y)),
            )

    def _end_drag(self, event: tk.Event) -> None:
        start = self._drag
        mode = self.navigation_mode.get()
        self._cancel_drag()
        if start is None or mode != "Rectangle zoom":
            return
        w, h = self._dimensions()
        x0, y0 = start
        x1 = min(w, max(0, event.x))
        y1 = min(h, max(0, event.y))
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            return
        self.view = rectangle_view(self.view, x0 / w, y0 / h, x1 / w, y1 / h)
        self._schedule_render()

    def _cancel_drag(self) -> None:
        self._drag = None
        if self._selection_id is not None:
            self.canvas.delete(self._selection_id)
            self._selection_id = None

    def _reset(self) -> None:
        self._cancel_drag()
        self.view = Viewport()
        self.request_render()

    def _close(self) -> None:
        self._closed = True
        self._cancel_drag()
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        if self._refine_pending is not None:
            self.root.after_cancel(self._refine_pending)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
