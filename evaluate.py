from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPE detector")
    parser.add_argument("--weights", default="runs/train/ppe_yolo/weights/best.pt")
    parser.add_argument("--data", default="data/ppe.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, plots=True)
    print(metrics)


if __name__ == "__main__":
    main()
