from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO PPE model")
    parser.add_argument("--weights", default="runs/train/ppe_yolo/weights/best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript", "tflite"])
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    output = model.export(format=args.format, imgsz=args.imgsz)
    print(f"Exported model: {output}")


if __name__ == "__main__":
    main()
