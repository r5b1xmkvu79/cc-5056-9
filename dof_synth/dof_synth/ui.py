"""Interactive viewer: drag the focus plane and inspect every mask.

Controls
--------
* Focus slider -- continuous focus distance in metres.
* Click / drag inside the DEPTH panel -- snaps the focus plane to the
  depth of the pixel under the cursor (a directly draggable focus plane).
* Aperture slider -- f-number; the CoC scales inversely with it.
* Method buttons -- forward scatter vs reverse layered gather.
* Mask buttons -- composite output, signed CoC, foreground footprint,
  edge-contamination, unknown background, unrecoverable range, in-focus,
  boundary layer, and the occlusion-ghost diagnosis overlay.
* Diagnose button -- runs the in-kernel trace on the pixels around the
  strongest occlusion edge, records how each layer contributes to the
  output pixel, tracks the DOF boundaries, and shows the per-layer
  defocus table plus the cause/stage of any duplicate contour.
* Save button -- writes the current 16-bit PNG output and mask array.

The raw colour/depth arrays are retained for the whole session; moving
the focus plane reruns the full solver from those raw measurements, so
the diagnosis is always recomputed from raw data rather than cached.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraParams
from .diagnostics import diagnose_occlusion, track_dof_boundaries
from .gather import render_gather
from .scatter import render_scatter
from .trace import TraceRequest

_MASK_VIEWS = [
    ("composite", None),
    ("signed CoC (px)", "signed_coc_px"),
    ("foreground", "foreground"),
    ("edge contaminated", "edge_contaminated"),
    ("unknown background", "unknown_background"),
    ("unrecoverable", "unrecoverable"),
    ("in focus", "in_focus"),
    ("bokeh radius (px)", "radius_px"),
    ("occlusion diagnosis", "diagnosis"),
]

_CAUSE_COLOURS = {
    "layer_fragmentation": np.array([1.0, 0.85, 0.1]),   # amber
    "kernel_coverage": np.array([1.0, 0.3, 0.2]),        # red
    "depth_noise": np.array([0.3, 0.7, 1.0]),            # blue
    "composite_order": np.array([0.8, 0.3, 1.0]),        # violet
    "normalization": np.array([0.2, 1.0, 0.6]),          # green
    "none": np.array([0.6, 0.6, 0.6]),                   # grey
}


class DoFViewer:
    def __init__(
        self,
        color: np.ndarray,
        depth_m: np.ndarray,
        camera: CameraParams,
        silhouette_m: np.ndarray | None = None,
        output_prefix: str = "dof_output",
    ):
        import matplotlib
        from matplotlib import pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        self._plt = plt
        self._cm = matplotlib.cm
        self.color = color
        self.depth_m = depth_m
        self.camera = camera
        self.silhouette = silhouette_m
        self.output_prefix = output_prefix
        self.method = "scatter"
        self.mask_view = "composite"
        self.result = None
        self.diagnostic = None       # last OcclusionDiagnostic (or None)
        self.diagnosis_note = ""     # human-readable status / failure text
        self.last_trace = None

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.fig.subplots_adjust(left=0.06, right=0.70, top=0.93, bottom=0.20)
        slider_ax_f = self.fig.add_axes([0.06, 0.10, 0.56, 0.03])
        slider_ax_n = self.fig.add_axes([0.06, 0.05, 0.56, 0.03])
        dmin, dmax = float(np.min(depth_m)), float(np.max(depth_m))
        if dmax <= dmin:
            dmin, dmax = dmin * 0.5, dmin * 2.0 + 1e-6
        margin = 0.2 * (dmax - dmin)
        self.focus_slider = Slider(
            slider_ax_f,
            "focus (m)",
            dmin - margin,
            dmax + margin,
            valinit=camera.focus_distance_m,
        )
        self.aperture_slider = Slider(
            slider_ax_n,
            "f-number",
            1.0,
            16.0,
            valinit=camera.f_number,
            valstep=0.25,
        )
        method_ax = self.fig.add_axes([0.74, 0.78, 0.24, 0.12])
        view_ax = self.fig.add_axes([0.74, 0.38, 0.24, 0.36])
        diag_ax = self.fig.add_axes([0.74, 0.30, 0.24, 0.06])
        save_ax = self.fig.add_axes([0.74, 0.22, 0.24, 0.05])
        self.info_ax = self.fig.add_axes([0.74, 0.02, 0.24, 0.18])
        self.info_ax.axis("off")

        self.method_buttons = RadioButtons(
            method_ax, ("scatter (near→far)", "gather (reverse)")
        )
        self.view_buttons = RadioButtons(
            view_ax, [name for name, _ in _MASK_VIEWS]
        )
        self.save_button = Button(save_ax, "Save output + masks")
        self.diagnose_button = Button(diag_ax, "Diagnose occlusion")

        self.focus_slider.on_changed(self._on_change)
        self.aperture_slider.on_changed(self._on_change)
        self.method_buttons.on_clicked(self._on_method)
        self.view_buttons.on_clicked(self._on_view)
        self.save_button.on_clicked(self._on_save)
        self.diagnose_button.on_clicked(self._on_diagnose)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click_depth)
        self._im = None
        self._render()
        self._draw_info()

    # -- solver ---------------------------------------------------------
    def _camera_now(self) -> CameraParams:
        return CameraParams(
            self.camera.f_mm,
            self.aperture_slider.val,
            self.focus_slider.val,
            self.camera.sensor_width_mm,
            self.camera.image_width_px,
            self.camera.image_height_px,
        )

    def _solver(self, camera: CameraParams, trace_request: TraceRequest | None):
        kwargs = dict(silhouette_m=self.silhouette, trace=trace_request)
        if self.method == "scatter":
            return render_scatter(self.color, self.depth_m, camera, **kwargs)
        return render_gather(self.color, self.depth_m, camera, **kwargs)

    def _render(self) -> None:
        camera = self._camera_now()
        self.result = self._solver(camera, None)
        # A focus or aperture change invalidates any prior diagnosis; the
        # next "Diagnose" reruns the trace against the new raw solve.
        self.diagnostic = None
        self.diagnosis_note = ""
        self._draw()
        self._draw_info()

    # -- ghost-edge discovery ------------------------------------------
    def _select_trace_pixels(
        self, camera: CameraParams, budget: int = 64
    ) -> tuple[list[tuple[int, int]], np.ndarray | None]:
        """Pick pixels straddling the strongest occlusion edge.

        Uses the depth-boundary tracker (never a default) to find edges,
        ranks them by foreground CoC, and returns both a small set of
        sample pixels and the occluder footprint used for ghost scoring.
        Returns ``([], None)`` when the frame has no usable edge.
        """
        report = track_dof_boundaries(
            self.depth_m, camera, coc_limit_px=1.0,
            depth_step_m=self._depth_step(),
        )
        if not report.boundaries:
            return [], None
        occluder = (self.depth_m < np.median(self.depth_m))
        if self.silhouette is not None:
            occluder = np.asarray(self.silhouette, dtype=bool)

        ranked = sorted(
            report.boundaries,
            key=lambda b: abs(b.near_coc_px),
            reverse=True,
        )
        h, w = self.depth_m.shape
        border = 2
        pixels: list[tuple[int, int]] = []
        seen = set()
        for b in ranked:
            # Skip boundaries on the frame border: the trace should sample
            # real interior occlusion edges, not convolution edge artifacts.
            # (Metrics elsewhere still cover the full frame.)
            if b.row < border or b.row >= h - border:
                continue
            if b.col < border or b.col >= w - border:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (int(np.clip(b.row + dr, 0, h - 1)),
                         int(np.clip(b.col + dc, 0, w - 1)))
                    if p not in seen:
                        seen.add(p)
                        pixels.append(p)
            if len(pixels) >= budget:
                break
        return pixels[:budget], occluder

    def _depth_step(self) -> float:
        return 0.15

    # -- diagnose -------------------------------------------------------
    def _on_diagnose(self, _event=None) -> None:
        camera = self._camera_now()
        pixels, occluder = self._select_trace_pixels(camera)
        if not pixels:
            self.diagnostic = None
            self.diagnosis_note = "No occlusion edge found to diagnose."
            self._draw_info()
            return
        request = TraceRequest(
            pixels=tuple(pixels),
            max_events_per_pixel=64,
            max_total_events=2048,
        )
        result = self._solver(camera, request)
        try:
            diag = diagnose_occlusion(
                result, self.depth_m, camera,
                occluder_mask=occluder,
                reference=None,           # no reference in the UI session
                trace=result.trace,
                depth_step_m=self._depth_step(),
            )
        except ValueError as exc:          # trace missing -> surface it
            self.diagnostic = None
            self.diagnosis_note = f"Diagnosis failed: {exc}"
            self._draw_info()
            return
        self.result = result
        self.diagnostic = diag
        note = (
            f"cause={diag.cause} @ stage={diag.stage} | "
            f"fragmented={diag.boundary_summary['fragmented_components']} comp"
        )
        if diag.trace_truncated:
            note += f" | trace truncated (+{diag.dropped_events} dropped)"
        self.diagnosis_note = note
        if self.mask_view == "occlusion diagnosis":
            self._draw()
        self._draw_info()

    # -- drawing --------------------------------------------------------
    def _draw(self) -> None:
        self.ax.clear()
        field = dict(_MASK_VIEWS)[self.mask_view]
        if field == "diagnosis":
            self._draw_diagnosis()
            return
        if field is None:
            img = np.clip(self.result.image, 0.0, 1.0)
            self._im = self.ax.imshow(img, interpolation="nearest")
        elif field in ("foreground", "edge_contaminated",
                       "unknown_background", "unrecoverable", "in_focus"):
            overlay = np.clip(self.result.image, 0.0, 1.0).copy()
            mask = getattr(self.result.masks, field)
            overlay[mask] = (
                overlay[mask] * 0.35
                + np.array([1.0, 0.15, 0.9]) * 0.65
            )
            self._im = self.ax.imshow(overlay, interpolation="nearest")
        else:
            data = getattr(self.result.masks, field)
            self._im = self.ax.imshow(
                data, cmap="coolwarm" if "signed" in field else "viridis",
                interpolation="nearest",
            )
        near, far = self.result.camera.dof_bounds_m(
            self.result.declaration["coc_limit_px"]
            * self.result.camera.pixel_pitch_mm
        )
        d = self.result.declaration
        self.ax.set_title(
            f"{self.result.method} | focus={self.result.camera.focus_distance_m:.3f} m"
            f" | f/{self.result.camera.f_number:g}\n"
            f"DoF [{near:.3f}, {'∞' if np.isinf(far) else f'{far:.3f}'}] m"
            f" | edge px={d['edge_pixels']}"
            f" unknown={d['unknown_pixels']}"
            f" unrecoverable={d['unrecoverable_pixels']}"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _draw_diagnosis(self) -> None:
        """Overlay the composite with occlusion edges coloured by cause."""
        self.ax.clear()
        if self.diagnostic is None:
            base = np.clip(self.result.image, 0.0, 1.0)
            self._im = self.ax.imshow(base, interpolation="nearest")
            self.ax.set_title(
                "occlusion diagnosis -- press 'Diagnose occlusion'",
                color="0.4",
            )
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            return
        diag = self.diagnostic
        # Fade the composite so the coloured evidence reads clearly.
        base = np.clip(self.result.image, 0.0, 1.0)
        faded = base * 0.45 + 0.2
        colour = _CAUSE_COLOURS.get(diag.cause, _CAUSE_COLOURS["none"])
        overlay = faded.copy()

        # Colour the fragmentation footprint (root evidence).
        frag = diag.fragmented_mask
        if frag.any():
            overlay[frag] = overlay[frag] * 0.4 + colour * 0.6

        # Mark the traced pixels with the stage that flagged them.
        for p in diag.per_pixel_stage:
            r, c = p["row"], p["col"]
            stage_colour = colour if p["stage"] == diag.stage else \
                _CAUSE_COLOURS["none"]
            r0, r1 = max(0, r - 1), min(overlay.shape[0], r + 2)
            c0, c1 = max(0, c - 1), min(overlay.shape[1], c + 2)
            overlay[r0:r1, c0:c1] = stage_colour

        self._im = self.ax.imshow(overlay, interpolation="nearest")
        d = self.result.declaration
        self.ax.set_title(
            f"occlusion diagnosis | cause={diag.cause} @ {diag.stage}\n"
            f"dim=fragmented surface, bright=traced pixels "
            f"(f/{self.result.camera.f_number:g}, "
            f"focus={self.result.camera.focus_distance_m:.3f} m)",
            fontsize=9.5,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _draw_info(self) -> None:
        """Per-layer defocus table + attribution log beside the canvas."""
        self.info_ax.clear()
        self.info_ax.axis("off")
        lines: list[str] = []
        if self.result is not None:
            d = self.result.declaration
            lines.append(
                f"layers={d.get('n_layers', '?')} "
                f"max_r={d.get('max_radius_px', 0):.2f}px"
            )
        if self.diagnosis_note:
            lines.append(self.diagnosis_note)
        if self.diagnostic is not None:
            diag = self.diagnostic
            lines.append("layer  r_px  depth_m  sign  provenance")
            for rec in diag.layer_records[:10]:
                prov = ""
                if rec["provenance"]:
                    prov = max(rec["provenance"].items(), key=lambda kv: kv[1])[0]
                lines.append(
                    f"{rec['layer']:5d} {rec['radius_px']:5.2f} "
                    f"{rec['min_depth_m']:6.3f} {rec['sign'][:4]:5s} {prov}"
                )
            if len(diag.layer_records) > 10:
                lines.append(f"... ({len(diag.layer_records)} layers)")
            lines.append("")
            lines.append("pixel        prov     fg_cov  bg_E  stage")
            for att in diag.attribution[:8]:
                lines.append(
                    f"({att.row:3d},{att.col:3d}) {att.provenance[:8]:8s} "
                    f"{att.foreground_coverage:5.2f} "
                    f"{att.background_energy:5.2f}"
                )
        elif not self.diagnosis_note:
            lines.append("Press 'Diagnose occlusion' to trace")
            lines.append("layer contributions and edges.")
        self.info_ax.text(
            0.0, 1.0, "\n".join(lines),
            family="monospace", fontsize=7.2, va="top", ha="left",
            transform=self.info_ax.transAxes,
        )
        self.fig.canvas.draw_idle()

    # -- events ---------------------------------------------------------
    def _on_change(self, _value: float) -> None:
        self._render()

    def _on_method(self, label: str) -> None:
        self.method = "scatter" if label.startswith("scatter") else "gather"
        self._render()

    def _on_view(self, label: str) -> None:
        self.mask_view = label
        self._draw()
        self._draw_info()

    def _on_click_depth(self, event) -> None:
        if event.inaxes is not self.ax or self._im is None:
            return
        if self.mask_view != "composite":
            return
        row, col = int(round(event.ydata)), int(round(event.xdata))
        h, w = self.depth_m.shape
        if 0 <= row < h and 0 <= col < w:
            self.focus_slider.set_val(float(self.depth_m[row, col]))

    def _on_save(self, _event) -> None:
        from matplotlib import image as mpimg

        base = self.output_prefix
        mpimg.imsave(
            f"{base}_{self.result.method}.png",
            np.clip(self.result.image, 0.0, 1.0),
        )
        np.savez_compressed(
            f"{base}_masks.npz",
            signed_coc_px=self.result.masks.signed_coc_px,
            radius_px=self.result.masks.radius_px,
            foreground=self.result.masks.foreground,
            edge_contaminated=self.result.masks.edge_contaminated,
            unknown_background=self.result.masks.unknown_background,
            unrecoverable=self.result.masks.unrecoverable,
            in_focus=self.result.masks.in_focus,
            provenance=self.result.masks.provenance,
            depth_m=self.depth_m,
            color=self.color,
        )

    def show(self) -> None:
        self._plt.show()


def launch(
    color: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraParams,
    silhouette_m: np.ndarray | None = None,
) -> DoFViewer:
    viewer = DoFViewer(color, depth_m, camera, silhouette_m)
    viewer.show()
    return viewer
