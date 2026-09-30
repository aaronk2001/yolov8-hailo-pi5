import numpy as np

from utils.postprocess import decode_yolov8

N_CLASSES = 80


def nms_output(boxes_by_class):
    """Build one Hailo NMS output: batch of 1 -> per-class arrays of [y0, x0, y1, x1, score]."""
    per_class = [np.zeros((0, 5), dtype=np.float32) for _ in range(N_CLASSES)]
    for class_id, rows in boxes_by_class.items():
        per_class[class_id] = np.array(rows, dtype=np.float32)
    return {'yolov8n/yolov8_nms_postprocess': [per_class]}


def test_empty_output_gives_no_detections():
    assert decode_yolov8(nms_output({})) == []


def test_reorders_hailo_yxyx_to_xyxy():
    raw = nms_output({5: [[0.1, 0.2, 0.7, 0.9, 0.88]]})
    (det,) = decode_yolov8(raw)
    assert det['class_id'] == 5
    assert np.allclose(det['bbox'], [0.2, 0.1, 0.9, 0.7])
    assert det['score'] == np.float32(0.88)


def test_confidence_threshold_is_inclusive():
    raw = nms_output({0: [[0, 0, 1, 1, 0.5], [0, 0, 1, 1, 0.49]]})
    scores = [d['score'] for d in decode_yolov8(raw, confidence_threshold=0.5)]
    assert scores == [0.5]


def test_keeps_class_ids_across_classes():
    raw = nms_output({0: [[0, 0, 0.5, 0.5, 0.9]], 79: [[0.5, 0.5, 1, 1, 0.8]]})
    assert [d['class_id'] for d in decode_yolov8(raw)] == [0, 79]
