import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
from supervision import ByteTrack

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from hailo_inference import HailoInference  # noqa: E402
from utils.draw import draw_detections  # noqa: E402
from utils.postprocess import decode_yolov8  # noqa: E402
from utils.tracking import to_sv_detections  # noqa: E402

parser = argparse.ArgumentParser(description='Live camera -> Hailo-8L YOLOv8 -> ByteTrack pipeline.')
parser.add_argument('--model', default='models/yolov8n.hef')
parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
parser.add_argument('--source', choices=['auto', 'csi', 'usb'], default='auto',
                    help='auto = Pi CSI camera if one is attached, else USB webcam')
parser.add_argument('--device', type=int, default=0, help='USB webcam index (/dev/videoN)')
parser.add_argument('--headless', action='store_true', help='no display window')
parser.add_argument('--duration', type=float, default=0, help='stop after N seconds (0 = until Q)')
parser.add_argument('--record', help='write annotated frames to this .mp4')
args = parser.parse_args()

# --- Shared state ---
frame_lock    = threading.Lock()
result_cv     = threading.Condition()
latest_frame  = None   # (seq, bgr)
latest_result = None   # (seq, bgr, dets)
running       = True
counts        = {'camera': 0, 'inference': 0}


def csi_available():
    try:
        from picamera2 import Picamera2
        return len(Picamera2.global_camera_info()) > 0
    except Exception:
        return False


def publish_frame(seq, bgr):
    global latest_frame
    with frame_lock:
        latest_frame = (seq, bgr)
    counts['camera'] += 1


# --- Camera thread ---
def csi_camera_thread():
    from picamera2 import Picamera2
    picam2 = Picamera2()
    mode = picam2.sensor_modes[0]
    config = picam2.create_video_configuration(
        sensor={'output_size': mode['size'], 'bit_depth': mode['bit_depth']},
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(3)
    seq = 0
    while running:
        frame = picam2.capture_array()
        seq += 1
        publish_frame(seq, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    picam2.stop()


def usb_camera_thread():
    global running
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open /dev/video{args.device}")
        running = False
        return
    seq = 0
    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        seq += 1
        publish_frame(seq, frame)
    cap.release()


# --- Inference thread: only runs on frames it hasn't seen yet ---
def inference_thread():
    global latest_result
    hailo = HailoInference(args.model)
    last_seq = 0
    while running:
        with frame_lock:
            item = latest_frame
        if item is None or item[0] == last_seq:
            time.sleep(0.001)
            continue
        last_seq, frame = item
        raw = hailo.run(hailo.preprocess(frame))
        dets = decode_yolov8(raw, confidence_threshold=args.conf)
        counts['inference'] += 1
        with result_cv:
            latest_result = (last_seq, frame, dets)
            result_cv.notify()
    hailo.release()


use_csi = args.source == 'csi' or (args.source == 'auto' and csi_available())
print(f"[INFO] Source: {'Pi CSI camera' if use_csi else f'USB webcam /dev/video{args.device}'}")
t_cam   = threading.Thread(target=csi_camera_thread if use_csi else usb_camera_thread, daemon=True)
t_infer = threading.Thread(target=inference_thread, daemon=True)
t_cam.start()
t_infer.start()

# --- ByteTracker setup ---
tracker = ByteTrack()
writer = None

# --- Main loop: one tracker update per new inference result ---
print("[INFO] Running" + ("" if args.headless else " - press Q to quit"))
t_start = time.time()
log_t, log_counts = t_start, dict(counts)
fps = 0.0
last_seq = 0

while running:
    with result_cv:
        result_cv.wait_for(lambda: latest_result is not None and latest_result[0] != last_seq  # noqa: B023
                           or not running, timeout=0.5)
        item = latest_result
    now = time.time()
    if args.duration and now - t_start >= args.duration:
        break
    if item is None or item[0] == last_seq:
        continue
    last_seq, frame, dets = item

    h, w = frame.shape[:2]
    tracked = tracker.update_with_detections(to_sv_detections(dets, w, h))

    # Draw detections
    annotated = draw_detections(frame.copy(), dets)

    # Draw tracker IDs
    if tracked.tracker_id is not None:
        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # FPS over the last 5 s: camera frames delivered vs frames inferred
    if now - log_t >= 5:
        dt = now - log_t
        cam_fps = (counts['camera'] - log_counts['camera']) / dt
        fps = (counts['inference'] - log_counts['inference']) / dt
        print(f"[FPS] camera={cam_fps:.1f} inference={fps:.1f}")
        log_t, log_counts = now, dict(counts)

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if args.record:
        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, h))
        writer.write(annotated)

    if not args.headless:
        cv2.imshow("YOLOv8n - Hailo-8L", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

running = False
elapsed = time.time() - t_start
t_infer.join(timeout=5)
t_cam.join(timeout=5)
if writer is not None:
    writer.release()
if not args.headless:
    cv2.destroyAllWindows()
print(f"[SUMMARY] {elapsed:.1f} s: camera {counts['camera']} frames ({counts['camera']/elapsed:.1f} FPS), "
      f"inference {counts['inference']} frames ({counts['inference']/elapsed:.1f} FPS)")
