"""Interactive application entry point.

Runs the procedural validation scene through the DoF viewer so the focus
plane can be dragged, the aperture changed, both solvers compared and
every computed mask inspected.  Run with ``python -m dof_synth.app``.
"""

from __future__ import annotations

from .camera import CameraParams
from .reference import render_pinhole
from .scene import analytic_masks, build_validation_scene
from .ui import launch


def main() -> None:
    size = 256
    camera = CameraParams(
        f_mm=35.0,
        f_number=1.2,
        focus_distance_m=2.75,
        sensor_width_mm=36.0,
        image_width_px=size,
        image_height_px=size,
    )
    scene = build_validation_scene()
    color, depth = render_pinhole(scene, camera)
    meta = analytic_masks(scene, camera, depth)
    launch(color, depth, camera, silhouette_m=meta.thin_slab_mask)


if __name__ == "__main__":
    main()
