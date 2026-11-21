"""Tkinter explorer for the Arnold tongue parameter plane."""

from __future__ import annotations

import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageTk

from .animation import AnimationOptions, save_zoom_gif
from .bifurcation import (
    BifurcationAnimationOptions,
    BifurcationOptions,
    render_bifurcation,
    save_bifurcation_gif,
)
from .core import HAS_NUMBA, CircleMapSettings
from .explorer import EXPLORER_MODES, RecurrenceOptions, render_explorer_rgb
from .export import save_image, suggested_filename
from .locking import colorize_locking, locking_field
from .map_families import MAP_FAMILIES
from .palettes import PALETTES
from .reference_gallery import REFERENCE_PRESETS, get_reference_preset
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
        startup = get_reference_preset("06 · Rainbow Showcase · Classic winding")
        self.view = startup.view
        self.settings = CircleMapSettings(transient=startup.transient, samples=startup.samples)
        self.palette = tk.StringVar(value=startup.palette)
        self.display_mode = tk.StringVar(value=startup.mode)
        self.recurrence_options = startup.recurrence
        self.map_family = tk.StringVar(value=startup.family)
        self.gamma = tk.DoubleVar(value=startup.gamma)
        self.navigation_mode = tk.StringVar(value="Rectangle zoom")
        self.transient = tk.StringVar(value=str(startup.transient))
        self.samples = tk.StringVar(value=str(startup.samples))
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
        ttk.Label(controls, text="Map").pack(side=tk.LEFT, padx=(10, 2))
        ttk.OptionMenu(
            controls,
            self.map_family,
            "Sine",
            *MAP_FAMILIES,
            command=lambda _: self._change_map_family(),
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Mode").pack(side=tk.LEFT, padx=(10, 2))
        ttk.OptionMenu(
            controls,
            self.display_mode,
            "Classic winding",
            "Mode locking",
            "Rotation number",
            *EXPLORER_MODES,
            command=lambda _: self._change_display_mode(),
        ).pack(side=tk.LEFT)
        ttk.Button(controls, text="Bifurcation…", command=self._open_bifurcation).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(controls, text="Gallery…", command=self._open_reference_gallery).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(controls, text="Recurrence…", command=self._set_recurrence).pack(
            side=tk.LEFT, padx=3
        )
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
        self.gamma_scale.state(["!disabled"])
        ttk.Label(controls, text="Transient").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.transient, width=6).pack(side=tk.LEFT)
        ttk.Label(controls, text="Samples").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(controls, textvariable=self.samples, width=6).pack(side=tk.LEFT)
        ttk.Button(controls, text="Apply", command=self._apply_settings).pack(side=tk.LEFT, padx=5)
        # Keep exports on a dedicated row: they must remain accessible on narrow screens.
        exports = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        exports.pack(fill=tk.X)
        ttk.Button(exports, text="Export image", command=self._export_image).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(exports, text="Export GIF", command=self._export_gif).pack(side=tk.LEFT)

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

    def _change_map_family(self) -> None:
        # Locking classification currently relies on sine-specific derivatives.
        if self.map_family.get() != "Sine" and self.display_mode.get() == "Mode locking":
            self.display_mode.set("Rotation number")
        self._change_display_mode()

    def _change_display_mode(self) -> None:
        if self.map_family.get() != "Sine" and self.display_mode.get() == "Mode locking":
            self.display_mode.set("Rotation number")
        self.gamma_scale.state(
            ["disabled"] if self.display_mode.get() == "Mode locking" else ["!disabled"]
        )
        self.request_render()

    def _set_recurrence(self) -> None:
        tolerance = simpledialog.askfloat(
            "Recurrence tolerance",
            "Return distance (0–0.5):",
            parent=self.root,
            initialvalue=self.recurrence_options.tolerance,
            minvalue=0.000001,
            maxvalue=0.499999,
        )
        if tolerance is None:
            return
        steps = simpledialog.askinteger(
            "Recurrence steps",
            "Maximum return steps (2–10000):",
            parent=self.root,
            initialvalue=self.recurrence_options.max_steps,
            minvalue=2,
            maxvalue=10000,
        )
        if steps is None:
            return
        self.recurrence_options = RecurrenceOptions(tolerance, steps)
        self.request_render()

    def _open_reference_gallery(self) -> None:
        """Choose a reproducible reference-inspired setup, without overwriting user exports."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Reference Gallery · exploratory presets")
        dialog.geometry("650x340")
        dialog.transient(self.root)
        ttk.Label(
            dialog,
            text="Reference-inspired presets (not pixel-exact reconstructions)",
            padding=12,
        ).pack(anchor=tk.W)
        choice = tk.StringVar(value=REFERENCE_PRESETS[1].label)
        selector = ttk.Combobox(
            dialog,
            textvariable=choice,
            state="readonly",
            values=[preset.label for preset in REFERENCE_PRESETS],
            width=38,
        )
        selector.pack(anchor=tk.W, padx=12)
        description = tk.StringVar()
        ttk.Label(dialog, textvariable=description, wraplength=560, padding=12).pack(
            anchor=tk.W, fill=tk.X
        )

        def update_note(_event: object | None = None) -> None:
            description.set(get_reference_preset(choice.get()).note)

        def apply_preset() -> None:
            preset = get_reference_preset(choice.get())
            if preset.mode == "Bifurcation":
                dialog.destroy()
                self._open_bifurcation()
                return
            self.view = preset.view
            self.map_family.set(preset.family)
            self.display_mode.set(preset.mode)
            self.palette.set(preset.palette)
            self.gamma.set(preset.gamma)
            self.settings = CircleMapSettings(
                transient=preset.transient,
                samples=preset.samples,
                theta0=self.settings.theta0,
            )
            self.transient.set(str(preset.transient))
            self.samples.set(str(preset.samples))
            self.recurrence_options = preset.recurrence
            dialog.destroy()
            self._change_display_mode()

        selector.bind("<<ComboboxSelected>>", update_note)
        update_note()
        ttk.Button(dialog, text="Apply preset", command=apply_preset).pack(
            anchor=tk.W, padx=12, pady=8
        )

    def _open_bifurcation(self) -> None:
        """Independent bifurcation explorer with rectangle zoom and pan."""
        window = tk.Toplevel(self.root)
        window.title("Bifurcation Explorer · K vs phase")
        window.geometry("1000x700")
        panel = ttk.Frame(window, padding=5)
        panel.pack(fill=tk.X)
        omega = tk.StringVar(value="0.35")
        transient = tk.StringVar(value="400")
        samples = tk.StringVar(value="240")
        family = tk.StringVar(value=self.map_family.get())
        palette = tk.StringVar(value="Classic")
        navigation = tk.StringVar(value="Rectangle zoom")
        current = [BifurcationOptions()]
        image_ref: list[ImageTk.PhotoImage] = []
        last_image: list[Image.Image] = []
        last_params: list[tuple[BifurcationOptions, str, str]] = []
        jobs: queue.SimpleQueue = queue.SimpleQueue()
        generation = [0]
        closed = [False]
        drag: list[tuple[int, int]] = []
        selection: list[int] = []
        resize_job: list[str | None] = [None]
        gif_running = [False]
        status = tk.StringVar(value="Drag a rectangle to zoom · switch to Pan to move")

        def field(label: str, variable: tk.StringVar) -> None:
            ttk.Label(panel, text=label).pack(side=tk.LEFT, padx=(4, 1))
            ttk.Entry(panel, textvariable=variable, width=6).pack(side=tk.LEFT)

        field("Ω", omega)
        field("Transient", transient)
        field("Samples", samples)
        ttk.OptionMenu(panel, family, family.get(), *MAP_FAMILIES).pack(side=tk.LEFT, padx=3)
        ttk.OptionMenu(panel, palette, palette.get(), "Classic", *PALETTES).pack(
            side=tk.LEFT, padx=3
        )
        ttk.OptionMenu(panel, navigation, navigation.get(), "Rectangle zoom", "Pan").pack(
            side=tk.LEFT, padx=3
        )
        canvas = tk.Canvas(window, background="#060817", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(window, textvariable=status, padding=5).pack(fill=tk.X)

        def options_from_fields() -> BifurcationOptions:
            return replace(
                current[0],
                omega=float(omega.get()),
                transient=int(transient.get()),
                samples=int(samples.get()),
            )

        def draw() -> None:
            try:
                opts = options_from_fields()
                selected_family, selected_palette = family.get(), palette.get()
            except ValueError as exc:
                messagebox.showerror("Invalid bifurcation parameters", str(exc), parent=window)
                return
            current[0] = opts
            generation[0] += 1
            job = generation[0]
            width, height = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
            status.set("Computing bifurcation diagram…")

            def work() -> None:
                try:
                    result = Image.fromarray(
                        render_bifurcation(
                            width, height, opts, family=selected_family, palette=selected_palette
                        )
                    )
                    jobs.put((job, result, None, opts, selected_family, selected_palette))
                except Exception as exc:
                    jobs.put((job, None, str(exc), opts, selected_family, selected_palette))

            self._executor.submit(work)

        def poll() -> None:
            if closed[0]:
                return
            while not jobs.empty():
                item = jobs.get()
                if isinstance(item[0], str):
                    if item[0] == "gif-progress":
                        status.set(f"Rendering GIF frame {item[1]}/{item[2]}…")
                    else:
                        gif_running[0] = False
                        gif_button.state(["!disabled"])
                        if item[0] == "gif-done":
                            status.set(f"GIF saved: {item[1]}")
                        else:
                            status.set(f"GIF export failed: {item[1]}")
                            messagebox.showerror("GIF export failed", item[1], parent=window)
                    continue
                job, result, error, opts, selected_family, selected_palette = item
                if job != generation[0]:
                    continue
                if error is not None:
                    status.set(f"Bifurcation render failed: {error}")
                elif result is not None:
                    last_image[:] = [result]
                    last_params[:] = [(opts, selected_family, selected_palette)]
                    image_ref[:] = [ImageTk.PhotoImage(result)]
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor="nw", image=image_ref[0])
                    status.set(
                        f"K [{opts.k_min:.3f}, {opts.k_max:.3f}] · "
                        f"phase [{opts.phase_min:.3f}, {opts.phase_max:.3f}]"
                    )
            window.after(80, poll)

        def on_canvas_resize(event: tk.Event) -> None:
            """Debounce resize to re-render at actual canvas dimensions."""
            if closed[0] or event.width < 2 or event.height < 2:
                return
            if resize_job[0] is not None:
                window.after_cancel(resize_job[0])
            resize_job[0] = window.after(350, render_after_resize)

        def render_after_resize() -> None:
            resize_job[0] = None
            if not closed[0]:
                draw()

        def export_gif() -> None:
            if gif_running[0]:
                return
            try:
                opts = options_from_fields()
                start = simpledialog.askfloat(
                    "GIF · Omega start",
                    "Starting Ω:",
                    parent=window,
                    initialvalue=opts.omega - 0.15,
                )
                if start is None:
                    return
                end = simpledialog.askfloat(
                    "GIF · Omega end",
                    "Ending Ω:",
                    parent=window,
                    initialvalue=opts.omega + 0.15,
                )
                if end is None:
                    return
                count = simpledialog.askinteger(
                    "GIF · Frames",
                    "Number of frames (2–120):",
                    parent=window,
                    initialvalue=24,
                    minvalue=2,
                    maxvalue=120,
                )
                if count is None:
                    return
                animation = BifurcationAnimationOptions(
                    omega_start=start,
                    omega_end=end,
                    frames=count,
                )
            except ValueError as exc:
                messagebox.showerror("Invalid animation settings", str(exc), parent=window)
                return
            selected_family, selected_palette = family.get(), palette.get()
            filename = (
                f"bifurcation_{selected_family.lower()}_{selected_palette.lower()}"
                f"_omega{start:.3f}-{end:.3f}_{count}f.gif"
            )
            path = filedialog.asksaveasfilename(
                parent=window,
                initialfile=filename,
                defaultextension=".gif",
                filetypes=[("Animated GIF", "*.gif")],
            )
            if not path:
                return
            gif_running[0] = True
            gif_button.state(["disabled"])
            status.set("Rendering bifurcation GIF…")

            def work() -> None:
                try:
                    save_bifurcation_gif(
                        path,
                        opts,
                        animation,
                        family=selected_family,
                        palette=selected_palette,
                        on_progress=lambda i, n: jobs.put(("gif-progress", i, n)),
                    )
                    jobs.put(("gif-done", path))
                except Exception as exc:
                    jobs.put(("gif-error", str(exc)))

            self._executor.submit(work)

        def apply_rectangle(start: tuple[int, int], end: tuple[int, int]) -> None:
            width, height = canvas.winfo_width(), canvas.winfo_height()
            if width < 2 or height < 2:
                return
            x0, x1 = sorted((max(0, min(width, start[0])), max(0, min(width, end[0]))))
            y0, y1 = sorted((max(0, min(height, start[1])), max(0, min(height, end[1]))))
            if x1 - x0 < 5 or y1 - y0 < 5:
                return
            opts = current[0]
            kspan = opts.k_max - opts.k_min
            pspan = opts.phase_max - opts.phase_min
            current[0] = replace(
                opts,
                k_min=opts.k_min + kspan * x0 / width,
                k_max=opts.k_min + kspan * x1 / width,
                phase_min=opts.phase_max - pspan * y1 / height,
                phase_max=opts.phase_max - pspan * y0 / height,
            )
            draw()

        def mouse_down(event: tk.Event) -> None:
            drag[:] = [(event.x, event.y)]
            if selection:
                canvas.delete(selection.pop())

        def mouse_move(event: tk.Event) -> None:
            if not drag or navigation.get() != "Rectangle zoom":
                return
            if selection:
                canvas.delete(selection.pop())
            selection.append(
                canvas.create_rectangle(
                    *drag[0], event.x, event.y, outline="white", dash=(4, 3), width=2
                )
            )

        def mouse_up(event: tk.Event) -> None:
            if not drag:
                return
            start = drag.pop()
            if selection:
                canvas.delete(selection.pop())
            if navigation.get() == "Rectangle zoom":
                apply_rectangle(start, (event.x, event.y))
                return
            width, height = canvas.winfo_width(), canvas.winfo_height()
            opts = current[0]
            dk = (event.x - start[0]) / width * (opts.k_max - opts.k_min)
            dp = (event.y - start[1]) / height * (opts.phase_max - opts.phase_min)
            kmin, kmax = opts.k_min - dk, opts.k_max - dk
            pmin, pmax = opts.phase_min + dp, opts.phase_max + dp
            # Keep the phase viewport within [0, 1].
            if pmin < 0:
                pmax -= pmin
                pmin = 0
            if pmax > 1:
                pmin -= pmax - 1
                pmax = 1
            current[0] = replace(opts, k_min=kmin, k_max=kmax, phase_min=pmin, phase_max=pmax)
            draw()

        def reset() -> None:
            current[0] = BifurcationOptions(
                omega=float(omega.get()), transient=int(transient.get()), samples=int(samples.get())
            )
            draw()

        def preset(name: str) -> None:
            presets = {
                "Overview": (0.35, 0.0, 4.0),
                "Fine structure": (0.35, 1.7, 2.5),
                "High coupling": (0.35, 2.7, 4.0),
            }
            om, lo, hi = presets[name]
            omega.set(str(om))
            current[0] = replace(current[0], k_min=lo, k_max=hi, phase_min=0.0, phase_max=1.0)
            draw()

        def export() -> None:
            if not last_params:
                messagebox.showinfo("Not ready", "Render a diagram first", parent=window)
                return
            opts, rendered_family, rendered_palette = last_params[0]
            filename = (
                f"bifurcation_{rendered_family.lower()}_{rendered_palette.lower()}"
                f"_omega{opts.omega:.3f}_k{opts.k_min:.3f}-{opts.k_max:.3f}"
                f"_phase{opts.phase_min:.3f}-{opts.phase_max:.3f}.png"
            )
            path = filedialog.asksaveasfilename(
                parent=window,
                initialfile=filename,
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            )
            if not path:
                return
            try:
                # Re-render at export resolution, not merely upscale the preview.
                opts, rendered_family, rendered_palette = last_params[0]
                image = Image.fromarray(
                    render_bifurcation(
                        1920, 1080, opts, family=rendered_family, palette=rendered_palette
                    )
                )
                image.save(path)
                status.set(f"Saved 1920 × 1080: {path}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc), parent=window)

        ttk.Button(panel, text="Render", command=draw).pack(side=tk.LEFT, padx=3)
        ttk.Button(panel, text="Reset", command=reset).pack(side=tk.LEFT)
        ttk.Button(panel, text="Export HD", command=export).pack(side=tk.LEFT, padx=3)
        gif_button = ttk.Button(panel, text="Export GIF", command=export_gif)
        gif_button.pack(side=tk.LEFT, padx=3)
        presets = ttk.Frame(window, padding=(6, 0, 6, 5))
        presets.pack(fill=tk.X, before=canvas)
        for name in ("Overview", "Fine structure", "High coupling"):
            ttk.Button(presets, text=name, command=lambda n=name: preset(n)).pack(
                side=tk.LEFT, padx=3
            )
        canvas.bind("<Configure>", on_canvas_resize)
        canvas.bind("<ButtonPress-1>", mouse_down)
        canvas.bind("<B1-Motion>", mouse_move)
        canvas.bind("<ButtonRelease-1>", mouse_up)

        def close() -> None:
            closed[0] = True
            generation[0] += 1
            if resize_job[0] is not None:
                window.after_cancel(resize_job[0])
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close)
        window.after(80, poll)
        window.after(200, draw)

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
        family = self.map_family.get()
        recurrence = self.recurrence_options
        self.status.set(
            f"Rendering preview…  Ω [{view.omega_min:.3f}, {view.omega_max:.3f}] "
            f"K [{view.coupling_min:.3f}, {view.coupling_max:.3f}] "
            f"Map={family} · Numba={'ON' if HAS_NUMBA else 'OFF'}"
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
                        renderer = (
                            render_explorer_rgb if display_mode in EXPLORER_MODES else render_rgb
                        )
                        rgb = renderer(
                            view,
                            rw,
                            rh,
                            settings,
                            palette=palette,
                            gamma=gamma,
                            family=family,
                            **(
                                {"mode": display_mode, "recurrence": recurrence}
                                if display_mode in EXPLORER_MODES
                                else {}
                            ),
                        )
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
        family = self.map_family.get()
        recurrence = self.recurrence_options
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
        filename = filename.removesuffix(".png") + f"_{family.lower()}"
        if display_mode in ("Poincare recurrence", "Recurrence order q"):
            filename += f"_eps{recurrence.tolerance:.4f}_max{recurrence.max_steps}"
        filename += ".png"
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
        family = self.map_family.get()
        self.status.set("Exporting image…")

        def work() -> None:
            try:
                if display_mode == "Mode locking":
                    rgb = colorize_locking(locking_field(view, width, height, settings), palette)
                else:
                    renderer = render_explorer_rgb if display_mode in EXPLORER_MODES else render_rgb
                    rgb = renderer(
                        view,
                        width,
                        height,
                        settings,
                        palette=palette,
                        gamma=gamma,
                        family=family,
                        **(
                            {"mode": display_mode, "recurrence": recurrence}
                            if display_mode in EXPLORER_MODES
                            else {}
                        ),
                    )
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
        family = self.map_family.get()
        recurrence = self.recurrence_options
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
            f"_{family.lower()}"
            f"_{options.motion.lower().replace(' ', '-')}_z{options.zoom_factor:.2f}"
            f"_{options.color_effect.lower().replace(' ', '-')}_"
            f"{options.second_palette.lower() if options.color_effect == 'Palette Pulse' else 'base'}"
            f"_seed{options.seed if options.motion == 'Random Walk' else 0}"
            f"_{options.frames}f.gif"
        )
        if mode in ("Poincare recurrence", "Recurrence order q"):
            filename = filename.removesuffix(".gif") + (
                f"_eps{recurrence.tolerance:.4f}_max{recurrence.max_steps}.gif"
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
                    family=family,
                    recurrence=recurrence,
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
