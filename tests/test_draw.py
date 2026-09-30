import numpy as np

from utils.draw import COLORS, draw_detections


def test_box_lands_on_xyxy_not_transposed():
    # Wide, short box in a non-square frame: a swapped-axes bug would draw it tall and narrow.
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    det = {'bbox': [0.1, 0.4, 0.9, 0.6], 'score': 0.9, 'class_id': 0}
    out = draw_detections(frame, [det])

    color = np.round(COLORS[0]).astype(np.uint8)
    assert (out[100, 40] == color).all()   # left edge, x0 = 0.1 * 400
    assert (out[100, 359] == color).all()  # right edge, x1 = 0.9 * 400
    assert not out[20, 200].any()          # far above the box: untouched


def test_draws_low_scores_passed_in():
    # The caller's --conf decides what's shown; draw must not re-filter.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(frame, [{'bbox': [0.2, 0.2, 0.8, 0.8], 'score': 0.25, 'class_id': 1}])
    assert out.any()
