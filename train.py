from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for construction PPE detection")
    parser.add_argument("--data", default="data/ppe.yaml", help="Path to YOLO dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model, e.g. yolo11n.pt, yolov8n.pt, yolo26n.pt if supported")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0 for GPU, cpu for CPU, or leave empty")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="ppe_yolo")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        optimizer="auto",
        plots=True,
        val=True,
    )

    print("Training completed.")
    print(f"Best weights are usually saved at: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
