"""End-to-end validation pipeline (also the headless test driver).

Renders the procedural scene with the pinhole camera, solves defocus
with both layered methods, renders the Monte-Carlo thin-lens reference
and the deliberately naive baseline, measures every failure mode and
writes actual pixel outputs plus the computed masks to ``outputs/``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from matplotlib import image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dof_synth.camera import CameraParams
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
