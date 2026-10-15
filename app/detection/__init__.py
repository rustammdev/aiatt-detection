# SPDX-License-Identifier: AGPL-3.0-only
"""Person detection and multi-object tracking.

This package is published on its own, under AGPL, and the rest of the product
is not (plan §13). That only works while it stays a separable work, so it
imports nothing of ours from outside itself: the detector takes plain values
rather than ``app.settings`` objects, and the private half depends on this
package and never the other way round.

``app.detection.yolo`` is deliberately not imported here, so reaching for the
box types or the tracker does not pull in ultralytics and torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

__all__ = ["PersonDetections", "build_tracker", "resolve_device"]

_LOGGER = logging.getLogger(__name__)

_BYTETRACK_REFERENCE_FPS = 30.0


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` to the best available device, warning on an unusable request."""
    import torch

    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested.startswith("cuda") and not available:
        _LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


@dataclass(frozen=True, slots=True)
class PersonDetections:
    """Boxes, plus the keypoints behind them when a pose model is loaded."""

    detections: sv.Detections
    keypoints: np.ndarray | None
    keypoint_confidence: np.ndarray | None

    @property
    def has_pose(self) -> bool:
        return self.keypoints is not None and len(self.keypoints) == len(self.detections)


def build_tracker(
    *,
    enabled: bool,
    activation_threshold: float,
    iou_threshold: float,
    minimum_consecutive_frames: int,
    lost_track_seconds: float,
    frame_rate: float,
) -> ByteTrackTracker | None:
    """Create a ByteTrack instance calibrated to the pipeline's effective frame rate."""
    if not enabled:
        return None
    # ByteTrack reads the buffer as frames at a 30 fps reference and rescales to
    # frame_rate itself, so a duration converts at 30 and not at the run's own
    # rate. The monolith passes seconds * frame_rate, which leaves the window at
    # seconds * frame_rate / 30: 1.7 s rather than 10 at target_fps 5.
    return ByteTrackTracker(
        track_activation_threshold=activation_threshold,
        minimum_iou_threshold=iou_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
        lost_track_buffer=max(1, round(lost_track_seconds * _BYTETRACK_REFERENCE_FPS)),
        frame_rate=max(1.0, frame_rate),
    )
