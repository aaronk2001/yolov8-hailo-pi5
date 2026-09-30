import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from hailo_inference import HailoInference  # noqa: E402
from utils.coco_labels import COCO_LABELS  # noqa: E402
from utils.draw import draw_detections  # noqa: E402
from utils.postprocess import decode_yolov8  # noqa: E402

parser = argparse.ArgumentParser(description='Run YOLOv8 on one image and save an annotated copy.')
parser.add_argument('--image', required=True)
parser.add_argument('--model', default='models/yolov8n.hef')
parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold')
parser.add_argument('--output', default='output.jpg')
args = parser.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    sys.exit(f"[ERROR] Cannot read image: {args.image}")
print(f"[INFO] Loaded: {frame.shape[1]}x{frame.shape[0]}")

hailo = HailoInference(args.model)
try:
    input_data = hailo.preprocess(frame)
    t0 = time.perf_counter()
    raw_outputs = hailo.run(input_data)
    print(f"[INFO] Inference: {(time.perf_counter() - t0) * 1000:.1f} ms")
finally:
    hailo.release()

detections = decode_yolov8(raw_outputs, confidence_threshold=args.conf)
print(f"[INFO] Detections: {len(detections)}")
for d in detections:
    print(f"  {COCO_LABELS[d['class_id']]:<12} score={d['score']:.2f} bbox={[round(x, 3) for x in d['bbox']]}")

cv2.imwrite(args.output, draw_detections(frame.copy(), detections))
print(f"[INFO] Saved: {args.output}")
