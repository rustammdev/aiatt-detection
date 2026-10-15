# SPDX-License-Identifier: AGPL-3.0-only
"""The YOLO detector. The one module in this service that touches ultralytics."""

from __future__ import annotations

import logging
import threading

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.detection import PersonDetections, resolve_device

__all__ = ["PersonDetector"]

_LOGGER = logging.getLogger(__name__)
_PERSON_CLASS_ID = 0


class PersonDetector:
    """YOLO wrapper restricted to the COCO ``person`` class.

    A ``*-pose`` checkpoint additionally yields keypoints, which is what lets the
    pipeline tell someone working at a desk from someone standing beside it.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str,
        confidence: float,
        iou: float,
        image_size: int,
        half_precision: bool,
    ) -> None:
        self._model_name = model
        self._confidence = confidence
        self._iou = iou
        self._image_size = image_size
        self.device = resolve_device(device)
        self._quantize = 16 if half_precision and self.device.startswith("cuda") else None
        self._model = YOLO(model)
        self._model.to(self.device)
        # Instances are cached and shared across pipeline runs; YOLO predict is
        # stateful enough that concurrent calls must be serialised.
        self._infer_lock = threading.Lock()
        _LOGGER.info("Loaded %s on %s (fp%s)", model, self.device, self._quantize or 32)

    @property
    def provides_pose(self) -> bool:
        return "pose" in self._model_name

    def detect(self, image: np.ndarray) -> PersonDetections:
        with self._infer_lock:
            result = self._model.predict(
                image,
                classes=[_PERSON_CLASS_ID],
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._image_size,
                quantize=self._quantize,
                verbose=False,
            )[0]
        detections = sv.Detections.from_ultralytics(result)

        keypoints = confidence = None
        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                confidence = result.keypoints.conf.cpu().numpy()
            else:
                confidence = np.ones(keypoints.shape[:2])
        return PersonDetections(detections, keypoints, confidence)
