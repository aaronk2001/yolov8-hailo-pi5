def decode_yolov8(raw_outputs: dict, confidence_threshold=0.4):
    """Flatten Hailo on-chip NMS output into a list of detections.

    Each output is batch -> per-class arrays of [y0, x0, y1, x1, score] rows,
    normalized to [0, 1]. Boxes are returned reordered as [x0, y0, x1, y1].
    """
    detections = []
    for batch in raw_outputs.values():
        for class_id, dets in enumerate(batch[0]):
            for row in dets:
                score = float(row[4])
                if score >= confidence_threshold:
                    detections.append({
                        'bbox': [float(row[1]), float(row[0]), float(row[3]), float(row[2])],
                        'score': score,
                        'class_id': class_id,
                    })
    return detections
