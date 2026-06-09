from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from .rules import Detection


class PPEDetector:
    """Thin wrapper around Ultralytics YOLO for PPE detection."""

    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.50, device: str | None = None):
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(self.weights)

    def predict_image(self, image: np.ndarray) -> List[Detection]:
        """Run YOLO on a BGR OpenCV image and return simplified detections."""
        results = self.model.predict(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        if not results:
            return []
        result = results[0]
        names = result.names
        detections: List[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            conf = float(box.conf[0].detach().cpu().item())
            detections.append(Detection(cls_name=str(names.get(cls_id, cls_id)), conf=conf, xyxy=tuple(xyxy)))
        return detections

    def predict_path(self, path: str | Path) -> tuple[np.ndarray, List[Detection]]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return image, self.predict_image(image)
