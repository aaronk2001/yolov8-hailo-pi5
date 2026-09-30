import numpy as np
from supervision import Detections


def to_sv_detections(dets, width, height):
    """Convert decode_yolov8 output (normalized x0, y0, x1, y1) to pixel-space supervision Detections."""
    if not dets:
        return Detections.empty()
    scale = np.array([width, height, width, height], dtype=np.float32)
    return Detections(
        xyxy=np.array([d['bbox'] for d in dets], dtype=np.float32) * scale,
        confidence=np.array([d['score'] for d in dets], dtype=np.float32),
        class_id=np.array([d['class_id'] for d in dets], dtype=int),
    )
