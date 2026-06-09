"""Create YOLO label files from an existing trained model.

Use this AFTER you already have a PPE model. It is for speeding up annotation:
1) Train a first model from a public/hand-labeled dataset.
2) Extract frames from your new videos.
3) Run this script to generate draft labels.
4) Manually review/fix labels before final training.

Usage:
    python scripts/pseudo_label.py --weights weights/best.pt --images datasets/ppe_raw/all/images --labels datasets/ppe_raw/all/labels --conf 0.45
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_line(cls_id: int, xyxy, width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Generate draft YOLO labels using an existing model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", default="datasets/ppe_raw/all/images")
    parser.add_argument("--labels", default="datasets/ppe_raw/all/labels")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    images = sorted([p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS])
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    written = 0
    skipped = 0
    for img in images:
        label_path = labels_dir / f"{img.stem}.txt"
        if label_path.exists() and not args.overwrite:
            skipped += 1
            continue
        result = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        h, w = result.orig_shape
        lines = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            line = yolo_line(cls_id, box.xyxy[0].tolist(), w, h)
            lines.append(line)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    print(f"Labels written: {written}")
    print(f"Skipped existing labels: {skipped}")
    print("IMPORTANT: pseudo-labels are draft labels. Review/fix them before final training.")


if __name__ == "__main__":
    main()
