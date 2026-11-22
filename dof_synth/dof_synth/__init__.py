"""Depth-of-field compositor from colour + metric depth inputs."""

from .camera import CameraParams
from .common import DoFMasks, DoFResult, prepare_inputs
from .diagnostics import (
    OcclusionDiagnostic,
    diagnose_occlusion,
    known_depth_step_probe,
    layer_fragmentation,
    measure_ghost,
    track_dof_boundaries,
)
from .gather import render_gather
from .scatter import render_scatter
from .trace import TraceRequest

__all__ = [
    "CameraParams",
    "DoFMasks",
    "DoFResult",
    "prepare_inputs",
    "render_scatter",
    "render_gather",
    "TraceRequest",
    "diagnose_occlusion",
    "known_depth_step_probe",
    "track_dof_boundaries",
    "layer_fragmentation",
    "measure_ghost",
    "OcclusionDiagnostic",
]
