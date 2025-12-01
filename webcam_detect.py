#!/usr/bin/env python3
"""
webcam_detect.py

Run real-time detection on your webcam using the project's YOLO weights.

Usage examples:
  # CPU-only, safe settings
  python webcam_detect.py --device cpu --imgsz 416 --conf 0.35

  # Use GPU 0, medium resolution
  python webcam_detect.py --device 0 --imgsz 640 --conf 0.25

Options:
  --model   Path to weights (default: runs/detect/train3/weights/best.pt if present else yolo11n.pt)
  --source  Webcam source (default: 0)
  --device  Device string: 'cpu' or GPU id like '0' (default: auto-detected)
  --imgsz   Inference image size (default: 640)
  --conf    Confidence threshold (default: 0.25)
  --show    Whether to display window (default: True)
  --save    Path to save the output video (optional)

This script uses the Ultralytics YOLO API and OpenCV to display boxes.
"""

import argparse
import os
import time

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("ultralytics package required. Install with: pip install ultralytics")

import cv2
import numpy as np


def detect_from_webcam(model_path: str, source: str, device: str, imgsz: int, conf: float, show: bool, save_path: str | None):
    print(f"Loading model: {model_path} on device={device}")
    model = YOLO(model_path)

    writer = None
    save_fps = 20

    stream = model.predict(source=source, device=device, imgsz=imgsz, conf=conf, stream=True)

    window_name = "Webcam Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for i, result in enumerate(stream):
            frame = getattr(result, 'orig_img', None)
            if frame is None:
                frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

            boxes = getattr(result, 'boxes', None)
            if boxes is not None:
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    try:
                        xyxy = boxes.xyxy.numpy()
                        confs = boxes.conf.numpy()
                        clss = boxes.cls.numpy().astype(int)
                    except Exception:
                        xyxy = []
                        confs = []
                        clss = []

                for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    try:
                        class_name = model.names[int(cl)]
                    except Exception:
                        class_name = str(int(cl))

                    conf_score = float(c)

                    label = f"{class_name} {conf_score:.2f}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # put label (with a filled rectangle background for readability)
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    y0 = max(0, y1 - t_size[1] - 6)
                    cv2.rectangle(frame, (x1, y0), (x1 + t_size[0] + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Quitting...')
                    break

            if save_path:
                h, w = frame.shape[:2]
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, save_fps, (w, h))
                writer.write(frame)

    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        print('Error while running detection:', e)
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run webcam detection using Ultralytics YOLO")
    parser.add_argument('--model', type=str, default=None, help='Path to model weights (.pt)')
    parser.add_argument('--source', type=str, default='0', help='Webcam source (0, 1, ... or path)')
    parser.add_argument('--device', type=str, default=None, help="Device: 'cpu' or GPU id like '0' (default: auto)")
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (px)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--show', action='store_true', help='Show output window')
    parser.add_argument('--save', type=str, default=None, help='Optional path to save output video (mp4)')

    args = parser.parse_args()

    default_model = 'runs/detect/train/weights/best.pt'
    if args.model is None:
        if os.path.exists(default_model):
            args.model = default_model
        else:
            args.model = 'yolo11m.pt' if os.path.exists('yolo11m.pt') else 'yolov8n.pt'

    if args.device is None:
        try:
            import torch
            args.device = '0' if torch.cuda.is_available() else 'cpu'
        except Exception:
            args.device = 'cpu'

    source = args.source
    if source.isdigit():
        source = int(source)

    show = True if args.show or args.save is None else True

    print('Starting webcam detection with settings:')
    print(f'  model: {args.model}')
    print(f'  source: {source}')
    print(f'  device: {args.device}')
    print(f'  imgsz: {args.imgsz}')
    print(f'  conf: {args.conf}')
    print(f'  show window: {show}')
    print(f'  save path: {args.save}')

    detect_from_webcam(args.model, source, args.device, args.imgsz, args.conf, show, args.save)
