import numpy as np
from supervision import ByteTrack

from utils.tracking import to_sv_detections


def det(bbox, score=0.9, class_id=0):
    return {'bbox': bbox, 'score': score, 'class_id': class_id}


def test_scales_normalized_boxes_to_pixels():
    sv = to_sv_detections([det([0.1, 0.2, 0.5, 0.8], 0.7, 5)], width=640, height=480)
    assert np.allclose(sv.xyxy, [[64, 96, 320, 384]])
    assert np.allclose(sv.confidence, [0.7])
    assert sv.class_id.tolist() == [5]


def test_no_detections_gives_empty():
    assert len(to_sv_detections([], 640, 480)) == 0


def test_bytetrack_keeps_id_for_a_moving_object():
    tracker = ByteTrack(minimum_consecutive_frames=1)
    ids = []
    for step in range(5):
        x = 0.1 + step * 0.01
        tracked = tracker.update_with_detections(to_sv_detections([det([x, 0.2, x + 0.3, 0.8])], 640, 480))
        ids.append(tracked.tracker_id.tolist())
    assert all(len(i) == 1 for i in ids)
    assert len({i[0] for i in ids}) == 1


def test_bytetrack_gives_separate_ids_to_separate_objects():
    tracker = ByteTrack(minimum_consecutive_frames=1)
    dets = [det([0.0, 0.0, 0.2, 0.2]), det([0.6, 0.6, 0.9, 0.9], class_id=2)]
    for _ in range(3):
        tracked = tracker.update_with_detections(to_sv_detections(dets, 640, 480))
    assert len(set(tracked.tracker_id.tolist())) == 2
