"""Optical sampling reference: Monte-Carlo thin-lens ray tracing.

The layered scatter/gather solvers are approximations of what a real
thin lens does.  This module renders the *same procedural scene* by
explicitly sampling the entrance pupil so the project has an
implementation-independent ground truth to compare against:

  * a ray starts at a uniformly sampled point on the disk-shaped pupil,
  * it is refracted through the lens toward the point where the pinhole
    (centre) ray meets the focus plane,
  * the scene is intersected and shaded,
  * averaging lens samples yields the physical defocus image including
    partial occlusion, bokeh of point lights and disoccluded regions.

A companion pinhole pass emits the colour/depth pair consumed by the
two compositing methods, guaranteeing identical camera conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraParams


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    color: np.ndarray
    emission: float = 0.0


@dataclass
class AABB:
    """Axis-aligned box used for the thin foreground slab."""

    lo: np.ndarray
    hi: np.ndarray
    color: np.ndarray
    emission: float = 0.0


@dataclass
class Scene:
    primitives: list
    background_color: np.ndarray


def _intersect_sphere(
    origins: np.ndarray, dirs: np.ndarray, sphere: Sphere
) -> tuple[np.ndarray, np.ndarray]:
    oc = origins - sphere.center
    b = np.einsum("ij,ij->i", oc, dirs)
    c = np.einsum("ij,ij->i", oc, oc) - sphere.radius**2
    disc = b * b - c
    hit = disc >= 0.0
    t = np.full(origins.shape[0], np.inf)
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t = np.where(t0 > 1e-4, t0, np.where(t1 > 1e-4, t1, np.inf))
    return t, hit & np.isfinite(t)


def _intersect_box(
    origins: np.ndarray, dirs: np.ndarray, box: AABB
) -> tuple[np.ndarray, np.ndarray]:
    inv = 1.0 / np.where(dirs == 0.0, 1e-12, dirs)
    t1 = (box.lo - origins) * inv
    t2 = (box.hi - origins) * inv
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    hit = (tmax >= np.maximum(tmin, 1e-4)) & (tmax > 1e-4)
    t = np.where(hit & (tmin > 1e-4), tmin, np.where(hit, tmax, np.inf))
    return t, hit


def _trace(
    origins: np.ndarray, dirs: np.ndarray, scene: Scene
) -> tuple[np.ndarray, np.ndarray]:
    n = origins.shape[0]
    best_t = np.full(n, np.inf)
    colors = np.tile(scene.background_color, (n, 1))
    for prim in scene.primitives:
        if isinstance(prim, Sphere):
            t, hit = _intersect_sphere(origins, dirs, prim)
        else:
            t, hit = _intersect_box(origins, dirs, prim)
        closer = hit & (t < best_t)
        best_t[closer] = t[closer]
        colors[closer] = prim.color + prim.emission
    return colors, best_t


def _ray_grid(camera: CameraParams) -> tuple[np.ndarray, np.ndarray]:
    h, w = camera.image_height_px, camera.image_width_px
    aspect = h / w
    f_px = camera.focal_length_px
    j = np.arange(w) + 0.5
    i = np.arange(h) + 0.5
    xx, yy = np.meshgrid(j, i)
    dx = (xx - w * 0.5) / f_px
    dy = (yy - h * 0.5) / f_px
    dz = np.ones_like(dx)
    dirs = np.stack([dx, dy * aspect, dz], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs.reshape(-1, 3), (h, w)


def render_pinhole(
    scene: Scene, camera: CameraParams
) -> tuple[np.ndarray, np.ndarray]:
    """Sharp centre-ray image and its metric depth map."""
    dirs, (h, w) = _ray_grid(camera)
    origins = np.zeros_like(dirs)
    color, t = _trace(origins, dirs, scene)
    # Metric depth along the optical axis, not ray length.
    depth = t * dirs[:, 2]
    return color.reshape(h, w, 3), depth.reshape(h, w)


def render_thin_lens(
    scene: Scene,
    camera: CameraParams,
    lens_samples: int = 96,
    seed: int = 7,
) -> np.ndarray:
    """Reference defocus image by Monte-Carlo pupil sampling."""
    rng = np.random.default_rng(seed)
    dirs, (h, w) = _ray_grid(camera)
    n = dirs.shape[0]
    f_m = camera.f_mm / 1000.0
    radius_m = camera.aperture_diameter_mm / 2000.0
    s = camera.focus_distance_m

    # Thin-lens geometry in metres: the lens plane is at z = f and the
    # sensor sits at the focus-image distance v(S) = f S / (S - f).
    v = f_m * s / (s - f_m)
    lens_z = v  # lens plane distance from sensor along +z
    accum = np.zeros((n, 3), dtype=np.float64)
    for _ in range(lens_samples):
        # Uniform disk sampling of the pupil (area-weighted).
        rho = np.sqrt(rng.random(n)) * radius_m
        theta = rng.random(n) * 2.0 * np.pi
        lens_pts = np.stack(
            [rho * np.cos(theta), rho * np.sin(theta), np.full(n, lens_z)],
            axis=-1,
        )
        # Point where the centre ray pierces the focus plane.
        focus_pts = dirs * (s / dirs[:, 2:3])
        ray_o = lens_pts
        ray_d = focus_pts - lens_pts
        ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)
        color, _ = _trace(ray_o, ray_d, scene)
        accum += color
    return (accum / lens_samples).reshape(h, w, 3)
