"""Thin-lens camera model.

Unit convention (enforced everywhere in this project):
  * distances in the scene / depth maps ......... meters
  * focal length, sensor size, aperture .......... millimetres
  * circle of confusion (CoC) on the sensor ...... millimetres
  * CoC expressed in pixels ...................... pixels (unitless count)

Projection convention:
  * camera at the origin looking along +Z
  * depth value of a pixel = distance z in metres along the optical axis
    of the pinhole (centre-ray) intersection
  * image coordinates: x right, y down, pixel centre at (i + 0.5, j + 0.5)

Signed CoC convention used throughout:
  coc_signed > 0  -> object is in FRONT of the focus plane (near blur)
  coc_signed < 0  -> object is BEHIND the focus plane (far blur)
  coc_signed = 0  -> object lies on the focus plane
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MM_PER_M = 1000.0


@dataclass(frozen=True)
class CameraParams:
    f_mm: float
    f_number: float
    focus_distance_m: float
    sensor_width_mm: float
    image_width_px: int
    image_height_px: int

    def __post_init__(self) -> None:
        if self.f_mm <= 0 or self.f_number <= 0:
            raise ValueError("focal length and f-number must be positive")
        if self.focus_distance_m <= self.f_mm / _MM_PER_M:
            raise ValueError("focus distance must exceed the focal length")
        if self.sensor_width_mm <= 0 or self.image_width_px <= 0:
            raise ValueError("sensor and image dimensions must be positive")

    @property
    def aperture_diameter_mm(self) -> float:
        """Entrance pupil diameter D = f / N."""
        return self.f_mm / self.f_number

    @property
    def pixel_pitch_mm(self) -> float:
        return self.sensor_width_mm / self.image_width_px

    @property
    def focal_length_px(self) -> float:
        return self.f_mm / self.pixel_pitch_mm

    def with_focus(self, focus_distance_m: float) -> "CameraParams":
        return CameraParams(
            self.f_mm,
            self.f_number,
            focus_distance_m,
            self.sensor_width_mm,
            self.image_width_px,
            self.image_height_px,
        )

    def with_f_number(self, f_number: float) -> "CameraParams":
        return CameraParams(
            self.f_mm,
            f_number,
            self.focus_distance_m,
            self.sensor_width_mm,
            self.image_width_px,
            self.image_height_px,
        )

    def signed_coc_mm(self, depth_m: np.ndarray | float) -> np.ndarray | float:
        """Signed circle-of-confusion diameter on the sensor in millimetres.

        Thin-lens derivation (object side):
            image distance of an object at z ........ v(z) = f z / (z - f)
            blur at the focus-plane sensor v(S):
                c = D * (v(S) - v(z)) / v(z)
                  = D f (S - z) / (z (S - f))
        Positive for z < S (foreground), negative for z > S (background).
        """
        z = np.asarray(depth_m, dtype=np.float64) * _MM_PER_M
        f = self.f_mm
        s = self.focus_distance_m * _MM_PER_M
        d = self.aperture_diameter_mm
        coc = d * f * (s - z) / (z * (s - f))
        if np.isscalar(depth_m):
            return float(coc)
        return coc

    def signed_coc_px(self, depth_m: np.ndarray | float) -> np.ndarray | float:
        coc_mm = self.signed_coc_mm(depth_m)
        if np.isscalar(depth_m):
            return coc_mm / self.pixel_pitch_mm
        return coc_mm / self.pixel_pitch_mm

    def coc_radius_px(self, depth_m: np.ndarray) -> np.ndarray:
        """Bokeh disk radius (half the CoC diameter), always >= 0."""
        return np.abs(self.signed_coc_px(depth_m)) * 0.5

    def hyperfocal_m(self, coc_limit_mm: float) -> float:
        """Hyperfocal distance H = f + f^2 / (N c)."""
        f = self.f_mm
        h_mm = f + f * f / (self.f_number * coc_limit_mm)
        return h_mm / _MM_PER_M

    def dof_bounds_m(self, coc_limit_mm: float) -> tuple[float, float]:
        """Near and far distances whose CoC equals coc_limit_mm.

        Returns (near, far); far is +inf when the focus is at/beyond
        the hyperfocal distance.
        """
        f = self.f_mm
        s = self.focus_distance_m * _MM_PER_M
        h = f + f * f / (self.f_number * coc_limit_mm)
        near_mm = h * s / (h + (s - f))
        denom = h - (s - f)
        if denom <= 0:
            far = np.inf
        else:
            far = (h * s / denom) / _MM_PER_M
        return near_mm / _MM_PER_M, far

    def infocus_mask(
        self, depth_m: np.ndarray, coc_limit_px: float = 1.0
    ) -> np.ndarray:
        return np.abs(self.signed_coc_px(depth_m)) <= coc_limit_px
