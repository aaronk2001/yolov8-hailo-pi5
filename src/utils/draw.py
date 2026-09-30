import cv2
import numpy as np

from .coco_labels import COCO_LABELS

COLORS = np.random.default_rng(42).uniform(0, 255, size=(len(COCO_LABELS), 3))


def draw_detections(frame, detections):
    h, w = frame.shape[:2]
    for det in detections:
        cls_id = int(det['class_id'])
        x0, y0, x1, y1 = det['bbox']  # normalized [0..1], from decode_yolov8
        x0, x1 = int(x0 * w), int(x1 * w)
        y0, y1 = int(y0 * h), int(y1 * h)
        color = COLORS[cls_id % len(COLORS)].tolist()
        label = f"{COCO_LABELS[cls_id]}: {det['score']:.2f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x0, y0 - th - 6), (x0 + tw, y0), color, -1)
        cv2.putText(frame, label, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame
