"""Procedurally generated validation scene.

The scene is built to stress every failure mode a naive
"blur by per-pixel radius" filter gets wrong:

  * a *thin* foreground slab (a few pixels wide) crossing the frame,
    with an analytic coverage mask so edge contamination is detectable;
  * an array of bright point lights in the background, whose total
    bokeh energy can be measured before and after defocus;
  * several matte layers at distinct depths so near/far blur, focus
    plane placement and occlusion ordering are all observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraParams
from .reference import AABB, Scene, Sphere


def build_validation_scene() -> Scene:
    bg = np.array([0.02, 0.025, 0.04])
    primitives = []

    # Far matte wall: a finite box covering the whole camera frustum,
    # so every pinhole ray hits a surface at a known metric depth.
    primitives.append(
        AABB(
            lo=np.array([-6.0, -6.0, 5.95]),
            hi=np.array([6.0, 6.0, 6.0]),
            color=np.array([0.18, 0.22, 0.30]),
        )
    )
    # Mid-depth block, behind the focus region and on the right so the
    # left half of the background stays visible for point-light tests.
    primitives.append(
        AABB(
            lo=np.array([0.85, -0.7, 2.6]),
            hi=np.array([1.55, 0.9, 2.9]),
            color=np.array([0.55, 0.35, 0.20]),
        )
    )
    # Near block, bottom-left corner (distinct depth layer; its projected
    # rows lie below the lowest point light so it never occludes one).
    primitives.append(
        AABB(
            lo=np.array([-1.55, -1.55, 1.4]),
            hi=np.array([-0.95, -1.25, 1.7]),
            color=np.array([0.20, 0.45, 0.35]),
        )
    )
    # Thin foreground slab: 6 cm deep, spanning the image vertically.
    # Its projected width is deliberately only a handful of pixels so a
    # per-pixel-radius blur would erase it; the solvers must keep its
    # full bokeh footprint.  It crosses the point-light array so halo
    # and occlusion-side energy are directly observable.
    primitives.append(
        AABB(
            lo=np.array([-0.18, -1.6, 0.70]),
            hi=np.array([-0.10, 1.6, 0.76]),
            color=np.array([0.85, 0.30, 0.25]),
        )
    )
    # Bright point lights: tiny emissive spheres in the background.
    # Inner columns sit only a few pixels from the slab silhouette so
    # their bokeh disks overlap the foreground footprint (observable
    # halo / occlusion-side energy); outer columns stay clear of blocks.
    grid = [
        # outer column, far left
        (-1.45, -0.95), (-1.40, 0.15), (-1.42, 1.15),
        # inner column, left flank of the slab (~2 px gap on screen)
        (-0.79, -0.85), (-0.79, 0.30), (-0.79, 1.10),
        # inner column, right flank of the slab (~2 px gap on screen)
        (-0.235, -0.65), (-0.235, 0.45), (-0.235, 1.20),
        # outer column, right (outside the mid block)
        (0.55, -1.05), (0.58, 0.15), (0.56, 1.05),
    ]
    for x, y in grid:
        primitives.append(
            Sphere(
                center=np.array([x, y, 3.05]),
                radius=0.022,
                color=np.array([0.0, 0.0, 0.0]),
                emission=6.0,
            )
        )
    return Scene(primitives, bg)


@dataclass
class SceneMeta:
    thin_slab_mask: np.ndarray      # analytic silhouette of the slab
    point_light_mask: np.ndarray    # pixels occupied by emissive spheres
    layer_depths_m: dict


def analytic_masks(
    scene: Scene,
    camera: CameraParams,
    depth_m: np.ndarray,
) -> SceneMeta:
    """Derive ground-truth structural masks from the pinhole depth."""
    h, w = depth_m.shape
    slab = min(
        (p for p in scene.primitives if isinstance(p, AABB)),
        key=lambda p: p.lo[2],
    )
    slab_mask = _project_box(slab, camera, h, w)

    point_mask = np.zeros((h, w), dtype=bool)
    for prim in scene.primitives:
        if isinstance(prim, Sphere) and prim.emission > 0:
            point_mask |= (
                np.abs(depth_m - prim.center[2]) < 0.05
            ) & _project_sphere(prim, camera, h, w)

    layers = {
        "thin_foreground": 0.73,
        "near_block": 1.55,
        "mid_block": 2.75,
        "point_lights": 3.05,
        "far_wall": 5.975,
    }
    return SceneMeta(slab_mask, point_mask, layers)


def _project_sphere(
    sphere: Sphere, camera: CameraParams, h: int, w: int
) -> np.ndarray:
    f_px = camera.focal_length_px
    z = sphere.center[2]
    cx_px = w * 0.5 + sphere.center[0] / z * f_px
    cy_px = h * 0.5 + sphere.center[1] / z * f_px / (w / h)
    r_px = sphere.radius / z * f_px
    j = np.arange(w) + 0.5
    i = np.arange(h) + 0.5
    xx, yy = np.meshgrid(j, i)
    return (xx - cx_px) ** 2 + (yy - cy_px) ** 2 <= r_px**2


def _project_box(
    box: AABB, camera: CameraParams, h: int, w: int
) -> np.ndarray:
    """Rasterise a box by projecting its 8 corners through the pinhole.

    The silhouette of a convex box is the convex hull of its projected
    corners; for an axis-aligned slab that hull is captured by the
    bounding rectangle of those projected points.
    """
    f_px = camera.focal_length_px
    aspect = h / w
    corners = np.array(
        [
            [x, y, z]
            for x in (box.lo[0], box.hi[0])
            for y in (box.lo[1], box.hi[1])
            for z in (box.lo[2], box.hi[2])
        ]
    )
    u = w * 0.5 + corners[:, 0] / corners[:, 2] * f_px
    v = h * 0.5 + corners[:, 1] / corners[:, 2] * f_px / aspect
    u0, u1 = np.floor(u.min()), np.ceil(u.max())
    v0, v1 = np.floor(v.min()), np.ceil(v.max())
    mask = np.zeros((h, w), dtype=bool)
    r0, r1 = max(int(v0), 0), min(int(v1) + 1, h)
    c0, c1 = max(int(u0), 0), min(int(u1) + 1, w)
    mask[r0:r1, c0:c1] = True
    return mask