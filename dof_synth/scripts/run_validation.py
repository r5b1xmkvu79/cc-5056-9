"""End-to-end validation pipeline (also the headless test driver).

Renders the procedural scene with the pinhole camera, solves defocus
with both layered methods, renders the Monte-Carlo thin-lens reference
and the deliberately naive baseline, measures every failure mode and
writes actual pixel outputs plus the computed masks to ``outputs/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from matplotlib import image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dof_synth.camera import CameraParams
from dof_synth.diagnostics import (
    diagnose_occlusion,
    known_depth_step_probe,
    track_dof_boundaries,
)
from dof_synth.gather import render_gather
from dof_synth.metrics import (
    energy_ratio,
    halo_energy,
    halo_excess,
    naive_radius_blur,
    penetration_error,
    per_pixel_error_map,
    rmse,
)
from dof_synth.reference import render_pinhole, render_thin_lens
from dof_synth.scene import analytic_masks, build_validation_scene
from dof_synth.scatter import render_scatter
from dof_synth.trace import TraceRequest


def _select_trace_pixels(
    depth: np.ndarray,
    camera: CameraParams,
    depth_step_m: float,
    budget: int = 32,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray | None]:
    """Pick interior pixels straddling the strongest occlusion edge.

    Uses the real boundary tracker; never substitutes a default when the
    frame has no edge (returns no pixels, and the caller records that).
    """
    report = track_dof_boundaries(
        depth, camera, coc_limit_px=1.0, depth_step_m=depth_step_m
    )
    if not report.boundaries:
        return (), None
    h, w = depth.shape
    border = 2
    ranked = sorted(report.boundaries, key=lambda b: abs(b.near_coc_px),
                    reverse=True)
    pixels: list[tuple[int, int]] = []
    seen = set()
    for b in ranked:
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
    return tuple(pixels[:budget]), (depth < np.median(depth))


def run(
    out_dir: str,
    size: int = 256,
    lens_samples: int = 96,
    n_layers: int = 48,
    seed: int = 7,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    h = w = size
    camera = CameraParams(
        f_mm=35.0,
        f_number=1.4,
        focus_distance_m=2.75,  # focus on the mid-depth block
        sensor_width_mm=36.0,
        image_width_px=w,
        image_height_px=h,
    )
    scene = build_validation_scene()
    color, depth = render_pinhole(scene, camera)
    meta = analytic_masks(scene, camera, depth)

    scatter = render_scatter(
        color, depth, camera, n_layers=n_layers,
        silhouette_m=meta.thin_slab_mask,
        depth_step_m=0.25,
    )
    gather = render_gather(
        color, depth, camera, n_layers=n_layers,
        silhouette_m=meta.thin_slab_mask,
        depth_step_m=0.25,
    )
    reference = np.clip(render_thin_lens(
        scene, camera, lens_samples=lens_samples, seed=seed
    ), 0.0, None)

    # --- occlusion diagnostics (bounded in-kernel trace) ---------------
    # The trace is taken from a *second* scatter solve with the same
    # parameters, over a small set of pixels straddling the strongest
    # interior occlusion edge.  Budgets bound the log cost but the drop /
    # truncation counters are kept as evidence.
    diag_depth_step = 0.25
    trace_pixels, occluder = _select_trace_pixels(
        depth, camera, diag_depth_step
    )
    trace_request = (
        TraceRequest(pixels=trace_pixels, max_events_per_pixel=96,
                     max_total_events=3072)
        if trace_pixels else None
    )
    traced = render_scatter(
        color, depth, camera, n_layers=n_layers,
        silhouette_m=meta.thin_slab_mask,
        depth_step_m=diag_depth_step,
        trace=trace_request,
    )
    if trace_request is not None:
        diagnostic = diagnose_occlusion(
            traced, depth, camera,
            occluder_mask=occluder,
            reference=reference,
            trace=traced.trace,
            depth_step_m=diag_depth_step,
        ).to_dict()
    else:
        diagnostic = {
            "cause": "no_boundary",
            "stage": "input",
            "note": "no interior occlusion edge found; nothing traced",
            "boundary_summary": track_dof_boundaries(
                depth, camera, coc_limit_px=1.0, depth_step_m=diag_depth_step
            ).summary,
        }
    # Known-depth-step control experiment, run through the same solvers.
    probe = known_depth_step_probe(
        camera, n_layers=n_layers, depth_step_m=diag_depth_step
    )
    diagnostics = {
        "occlusion": diagnostic,
        "known_depth_step": probe,
    }
    # Stable digest so a re-run can be checked for identical evidence.
    evidence_digest = hashlib.sha256(
        json.dumps(
            {
                "occlusion": {
                    k: diagnostic.get(k)
                    for k in ("cause", "stage", "stage_flags",
                              "boundary_summary", "per_pixel_stage",
                              "trace_truncated", "dropped_events")
                },
                "known_depth_step": {
                    k: probe.get(k)
                    for k in ("isolated_cause", "cause_thresholds",
                              "order_swap")
                },
            },
            sort_keys=True, default=float,
        ).encode()
    ).hexdigest()
    diagnostics["evidence_digest"] = evidence_digest

    radius = scatter.masks.radius_px
    naive = np.clip(naive_radius_blur(color, radius), 0.0, None)

    # --- metrics -------------------------------------------------------
    max_r = int(np.ceil(radius.max())) + 2
    point_energy = energy_ratio(color, scatter.image,
                                meta.point_light_mask, max_r)
    point_energy_g = energy_ratio(color, gather.image,
                                  meta.point_light_mask, max_r)

    # Point-support mask: where bokeh energy from the lights can land.
    from scipy import ndimage
    point_support = ndimage.binary_dilation(
        meta.point_light_mask, iterations=max_r
    )
    halo_dil = max(2, max_r // 2)
    halo_s = halo_energy(
        scatter.image, point_support, meta.thin_slab_mask, halo_dil
    )
    halo_g = halo_energy(
        gather.image, point_support, meta.thin_slab_mask, halo_dil
    )
    halo_ref = halo_energy(
        reference, point_support, meta.thin_slab_mask, halo_dil
    )
    footprint = ndimage.binary_dilation(
        meta.thin_slab_mask, iterations=halo_dil
    )
    pen_s = penetration_error(scatter.image, reference, footprint)
    pen_g = penetration_error(gather.image, reference, footprint)
    pen_naive = penetration_error(naive, reference, footprint)

    eval_mask = ~scatter.masks.unrecoverable
    metrics = {
        "camera": {
            "f_mm": camera.f_mm,
            "f_number": camera.f_number,
            "focus_m": camera.focus_distance_m,
            "pixel_pitch_mm": camera.pixel_pitch_mm,
            "max_coc_radius_px": float(radius.max()),
        },
        "rmse_scatter_vs_reference": rmse(scatter.image, reference, eval_mask),
        "rmse_gather_vs_reference": rmse(gather.image, reference, eval_mask),
        "rmse_scatter_vs_gather": rmse(scatter.image, gather.image),
        "rmse_naive_vs_reference": rmse(naive, reference, eval_mask),
        "rmse_scatter_on_thin_foreground_footprint": rmse(
            scatter.image, reference, footprint
        ),
        "rmse_gather_on_thin_foreground_footprint": rmse(
            gather.image, reference, footprint
        ),
        "rmse_naive_on_thin_foreground_footprint": rmse(
            naive, reference, footprint
        ),
        "point_light_energy_ratio_scatter": point_energy,
        "point_light_energy_ratio_gather": point_energy_g,
        "halo_energy_scatter": halo_s,
        "halo_energy_gather": halo_g,
        "halo_energy_reference": halo_ref,
        "halo_excess_scatter_vs_ref": halo_excess(
            scatter.image, reference, point_support,
            meta.thin_slab_mask, halo_dil,
        ),
        "halo_excess_gather_vs_ref": halo_excess(
            gather.image, reference, point_support,
            meta.thin_slab_mask, halo_dil,
        ),
        "penetration_scatter_vs_ref": pen_s,
        "penetration_gather_vs_ref": pen_g,
        "penetration_naive_vs_ref": pen_naive,
        "declaration": scatter.declaration,
        "dof_bounds_m": camera.dof_bounds_m(
            scatter.declaration["coc_limit_px"] * camera.pixel_pitch_mm
        ),
        "diagnostics": diagnostics,
    }

    # --- actual pixel outputs -----------------------------------------
    def save(img, name, clip=True):
        mpimg.imsave(
            os.path.join(out_dir, name),
            np.clip(img, 0.0, 1.0) if clip else img,
        )

    save(color, "00_pinhole_color.png")
    mpimg.imsave(os.path.join(out_dir, "01_depth.png"),
                 depth, cmap="viridis")
    save(reference, "02_reference_thin_lens.png")
    save(scatter.image, "03_scatter.png")
    save(gather.image, "04_gather.png")
    save(naive, "05_naive_radius_baseline.png")
    err = per_pixel_error_map(scatter.image, reference)
    mpimg.imsave(os.path.join(out_dir, "06_error_scatter_vs_ref.png"),
                 err / max(err.max(), 1e-6), cmap="magma")
    mpimg.imsave(
        os.path.join(out_dir, "07_signed_coc.png"),
        scatter.masks.signed_coc_px, cmap="coolwarm",
    )
    for name, mask in [
        ("08_mask_edge.png", scatter.masks.edge_contaminated),
        ("09_mask_foreground.png", scatter.masks.foreground),
        ("10_mask_unknown.png", scatter.masks.unknown_background),
        ("11_mask_unrecoverable.png", scatter.masks.unrecoverable),
        ("12_mask_infocus.png", scatter.masks.in_focus),
        ("13_mask_thin_slab_groundtruth.png", meta.thin_slab_mask),
        ("14_mask_point_lights.png", meta.point_light_mask),
    ]:
        mpimg.imsave(os.path.join(out_dir, name),
                     mask.astype(np.uint8) * 255, cmap="gray")

    np.savez_compressed(
        os.path.join(out_dir, "raw_inputs_and_outputs.npz"),
        color=color,
        depth_m=depth,
        reference=reference,
        scatter=scatter.image,
        gather=gather.image,
        naive=naive,
        signed_coc_px=scatter.masks.signed_coc_px,
        foreground=scatter.masks.foreground,
        edge=scatter.masks.edge_contaminated,
        unknown=scatter.masks.unknown_background,
        unrecoverable=scatter.masks.unrecoverable,
        slab=meta.thin_slab_mask,
        points=meta.point_light_mask,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lens-samples", type=int, default=96)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    ))
    args = parser.parse_args()
    result = run(args.out, args.size, args.lens_samples, args.layers)
    print(json.dumps(result, indent=2, default=float))
