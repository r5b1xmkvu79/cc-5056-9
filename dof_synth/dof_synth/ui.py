"""Interactive viewer: drag the focus plane and inspect every mask.

Controls
--------
* Focus slider -- continuous focus distance in metres.
* Click / drag inside the DEPTH panel -- snaps the focus plane to the
  depth of the pixel under the cursor (a directly draggable focus plane).
* Aperture slider -- f-number; the CoC scales inversely with it.
* Method buttons -- forward scatter vs reverse layered gather.
* Mask buttons -- composite output, signed CoC, foreground footprint,
  edge-contamination, unknown background, unrecoverable range, in-focus.
* Save button -- writes the current 16-bit PNG output and mask array.

The raw colour/depth arrays are retained for the whole session; moving
the focus plane reruns the full solver from those raw measurements.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraParams
from .gather import render_gather
from .scatter import render_scatter

_MASK_VIEWS = [
    ("composite", None),
    ("signed CoC (px)", "signed_coc_px"),
    ("foreground", "foreground"),
    ("edge contaminated", "edge_contaminated"),
    ("unknown background", "unknown_background"),
    ("unrecoverable", "unrecoverable"),
    ("in focus", "in_focus"),
    ("bokeh radius (px)", "radius_px"),
]


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

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(left=0.08, right=0.72, top=0.92, bottom=0.20)
        slider_ax_f = self.fig.add_axes([0.08, 0.10, 0.60, 0.03])
        slider_ax_n = self.fig.add_axes([0.08, 0.05, 0.60, 0.03])
        dmin, dmax = float(np.min(depth_m)), float(np.max(depth_m))
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
        method_ax = self.fig.add_axes([0.76, 0.62, 0.21, 0.16])
        view_ax = self.fig.add_axes([0.76, 0.22, 0.21, 0.36])
        save_ax = self.fig.add_axes([0.76, 0.13, 0.21, 0.05])

        self.method_buttons = RadioButtons(
            method_ax, ("scatter (near→far)", "gather (reverse)")
        )
        self.view_buttons = RadioButtons(
            view_ax, [name for name, _ in _MASK_VIEWS]
        )
        self.save_button = Button(save_ax, "Save output + masks")

        self.focus_slider.on_changed(self._on_change)
        self.aperture_slider.on_changed(self._on_change)
        self.method_buttons.on_clicked(self._on_method)
        self.view_buttons.on_clicked(self._on_view)
        self.save_button.on_clicked(self._on_save)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click_depth)
        self._im = None
        self._render()

    def _solver(self, camera: CameraParams):
        if self.method == "scatter":
            return render_scatter(
                self.color, self.depth_m, camera,
                silhouette_m=self.silhouette,
            )
        return render_gather(
            self.color, self.depth_m, camera,
            silhouette_m=self.silhouette,
        )

    def _render(self) -> None:
        camera = CameraParams(
            self.camera.f_mm,
            self.aperture_slider.val,
            self.focus_slider.val,
            self.camera.sensor_width_mm,
            self.camera.image_width_px,
            self.camera.image_height_px,
        )
        self.result = self._solver(camera)
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()
        field = dict(_MASK_VIEWS)[self.mask_view]
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

    def _on_change(self, _value: float) -> None:
        self._render()

    def _on_method(self, label: str) -> None:
        self.method = "scatter" if label.startswith("scatter") else "gather"
        self._render()

    def _on_view(self, label: str) -> None:
        self.mask_view = label
        self._draw()

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
