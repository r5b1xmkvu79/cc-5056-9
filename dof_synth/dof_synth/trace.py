"""Bounded, kernel-driven tracing of the layered defocus compositor.

Every diagnostic in this project must be produced by the *actual* solver
loop, not by a re-implementation that can silently drift from it.  This
module therefore defines the small transport objects that the solvers
(``scatter`` / ``gather``) fill in-place, and a recorder that caps how
much evidence is retained.

The recorder deliberately keeps *bounded* state: a bounded number of
pixels, a bounded number of contribution events per pixel and a global
event ceiling.  When a bound is hit the recorder increments a drop
counter and sets ``truncated`` instead of growing without limit.  The
counters are part of the evidence, so "we stopped recording" can never
be mistaken for "nothing was there" (no silent defaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Trust categories of the depth/colour sample feeding the blur estimate.
PROVENANCE_NAMES = (
    "background",     # 0: measured background sample
    "foreground",     # 1: measured foreground sample
    "edge",           # 2: silhouette boundary (depth may be polluted)
    "unknown_fill",   # 3: filled, no direct evidence (declared approximation)
    "unrecoverable",  # 4: filled beyond the declared radius (approximation)
)
PROVENANCE_BACKGROUND = 0
PROVENANCE_FOREGROUND = 1
PROVENANCE_EDGE = 2
PROVENANCE_UNKNOWN = 3
PROVENANCE_UNRECOVERABLE = 4

_EPS_COVERAGE = 1e-6
_EPS_ENERGY = 1e-9


@dataclass(frozen=True)
class TraceRequest:
    """Which pixels to trace and how much evidence to retain."""

    pixels: tuple[tuple[int, int], ...]
    provenance: np.ndarray | None = None
    max_events_per_pixel: int = 64
    max_total_events: int = 4096

    def __post_init__(self) -> None:
        if self.max_events_per_pixel <= 0 or self.max_total_events <= 0:
            raise ValueError("trace event budgets must be positive")


@dataclass
class ContributionEvent:
    """One layer's contribution to one tracked output pixel."""

    order_rank: int
    layer: int
    radius_px: float
    layer_sign: str          # 'foreground' | 'background' | 'focus' | 'mixed'
    layer_min_depth_m: float
    coverage: float          # blurred coverage alpha deposited by this layer
    weight: float            # compositing multiplier actually applied
    acc_before: float        # relevant accumulator value before this layer
    energy_rgb: tuple[float, float, float]  # premultiplied energy added


@dataclass
class PixelTrace:
    row: int
    col: int
    depth_m: float
    signed_coc_px: float
    radius_px: float
    layer_index: int
    provenance: str
    events: list[ContributionEvent] = field(default_factory=list)
    output_rgb: tuple[float, float, float] | None = None
    truncated: bool = False

    def energy_total(self) -> float:
        return float(sum(sum(e.energy_rgb) for e in self.events))


@dataclass
class LayerTrace:
    method: str
    n_layers: int
    layer_radius_px: np.ndarray
    layer_min_depth_m: np.ndarray
    layer_signs: dict[int, str]
    layer_provenance: dict[int, dict[str, int]]
    pixels: dict[tuple[int, int], PixelTrace]
    max_events_per_pixel: int
    max_total_events: int
    dropped_events: int
    truncated: bool

    def event_count(self) -> int:
        return sum(len(p.events) for p in self.pixels.values())

    def mixed_sign_layers(self) -> dict[int, str]:
        return {
            k: s for k, s in self.layer_signs.items() if s == "mixed"
        }


class TraceRecorder:
    """In-loop recorder used by the scatter/gather solvers."""

    def __init__(
        self,
        request: TraceRequest,
        *,
        provenance: np.ndarray,
        signed_coc_px: np.ndarray,
        depth_m: np.ndarray,
        layer_index: np.ndarray,
        layer_radius: np.ndarray,
        layer_min_depth: np.ndarray,
        method: str,
        n_layers: int,
    ) -> None:
        self.request = request
        self.provenance = provenance
        self.signed = signed_coc_px
        self.depth = depth_m
        self.layer_index = layer_index
        self.layer_radius = layer_radius
        self.layer_min_depth = layer_min_depth
        self.method = method
        self.n_layers = n_layers

        h, w = depth_m.shape
        seen: list[tuple[int, int]] = []
        for r, c in request.pixels:
            rr, cc = int(r), int(c)
            if not (0 <= rr < h and 0 <= cc < w):
                # Never drop a requested pixel silently.
                raise ValueError(
                    f"trace pixel ({r}, {c}) outside image {h}x{w}"
                )
            if (rr, cc) not in seen:
                seen.append((rr, cc))
        self.pixel_coords = seen
        self.rows = np.array([p[0] for p in seen], dtype=np.intp)
        self.cols = np.array([p[1] for p in seen], dtype=np.intp)

        self.pixels: dict[tuple[int, int], PixelTrace] = {}
        for r, c in seen:
            prov = int(provenance[r, c])
            signed = float(signed_coc_px[r, c])
            self.pixels[(r, c)] = PixelTrace(
                row=r,
                col=c,
                depth_m=float(depth_m[r, c]),
                signed_coc_px=signed,
                radius_px=abs(signed) * 0.5,
                layer_index=int(layer_index[r, c]),
                provenance=PROVENANCE_NAMES[prov]
                if 0 <= prov < len(PROVENANCE_NAMES)
                else f"code_{prov}",
            )

        self.layer_signs: dict[int, str] = {}
        self.layer_provenance: dict[int, dict[str, int]] = {}
        self.dropped_events = 0
        self.truncated = False
        self._events_total = 0

    def begin_layer(
        self, k: int, radius_px: float, order_rank: int, sel: np.ndarray
    ) -> None:
        """Record the source-sign and provenance mix of a layer."""
        if not sel.any():
            return
        signs = self.signed[sel]
        n_fg = int((signs > 0.0).sum())
        n_bg = int((signs < 0.0).sum())
        if n_fg and n_bg:
            sign = "mixed"
        elif n_fg:
            sign = "foreground"
        elif n_bg:
            sign = "background"
        else:
            sign = "focus"
        # A zero-radius layer cannot visibly occlude, so a sign mix there
        # is not an anomaly.
        if sign == "mixed" and radius_px <= _EPS_COVERAGE:
            sign = "focus"
        self.layer_signs[k] = sign

        codes, counts = np.unique(self.provenance[sel], return_counts=True)
        self.layer_provenance[k] = {
            (PROVENANCE_NAMES[int(c)]
             if 0 <= int(c) < len(PROVENANCE_NAMES) else f"code_{int(c)}"):
                int(n)
            for c, n in zip(codes, counts)
        }

    def note(
        self,
        k: int,
        radius_px: float,
        order_rank: int,
        coverage: np.ndarray,
        weight: np.ndarray,
        acc_before: np.ndarray,
        energy_rgb: np.ndarray,
    ) -> None:
        """Record this layer's contribution at every tracked pixel.

        ``coverage``/``weight``/``acc_before`` are full image-shaped
        arrays and ``energy_rgb`` is image-shaped + 3; only the tracked
        pixels are sampled, so recording stays O(tracked pixels), not
        O(image).
        """
        if not self.pixel_coords:
            return
        sign = self.layer_signs.get(k, "focus")
        min_depth = float(self.layer_min_depth[k])
        cov_vals = coverage[self.rows, self.cols]
        weight_vals = weight[self.rows, self.cols]
        acc_vals = acc_before[self.rows, self.cols]
        energy_vals = energy_rgb[self.rows, self.cols]
        for i, coord in enumerate(self.pixel_coords):
            pt = self.pixels[coord]
            cov = float(cov_vals[i])
            energy = energy_vals[i]
            if cov <= _EPS_COVERAGE and float(np.max(energy)) <= _EPS_ENERGY:
                continue
            if (
                len(pt.events) >= self.request.max_events_per_pixel
                or self._events_total >= self.request.max_total_events
            ):
                self.dropped_events += 1
                pt.truncated = True
                self.truncated = True
                continue
            pt.events.append(
                ContributionEvent(
                    order_rank=int(order_rank),
                    layer=int(k),
                    radius_px=float(radius_px),
                    layer_sign=sign,
                    layer_min_depth_m=min_depth,
                    coverage=cov,
                    weight=float(weight_vals[i]),
                    acc_before=float(acc_vals[i]),
                    energy_rgb=(
                        float(energy[0]),
                        float(energy[1]),
                        float(energy[2]),
                    ),
                )
            )
            self._events_total += 1

    def finalize(self, image: np.ndarray) -> None:
        for (r, c), pt in self.pixels.items():
            pt.output_rgb = (
                float(image[r, c, 0]),
                float(image[r, c, 1]),
                float(image[r, c, 2]),
            )

    def layer_trace(self) -> LayerTrace:
        return LayerTrace(
            method=self.method,
            n_layers=self.n_layers,
            layer_radius_px=np.array(self.layer_radius, dtype=np.float64),
            layer_min_depth_m=np.array(self.layer_min_depth, dtype=np.float64),
            layer_signs=dict(self.layer_signs),
            layer_provenance=dict(self.layer_provenance),
            pixels=self.pixels,
            max_events_per_pixel=self.request.max_events_per_pixel,
            max_total_events=self.request.max_total_events,
            dropped_events=self.dropped_events,
            truncated=self.truncated,
        )
