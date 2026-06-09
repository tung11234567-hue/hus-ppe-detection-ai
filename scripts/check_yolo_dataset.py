"""Check common YOLO dataset mistakes before training."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ppe.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    names = cfg["names"]
    if isinstance(names, dict):
        n_classes = len(names)
        class_names = {int(k): v for k, v in names.items()}
    else:
        n_classes = len(names)
        class_names = dict(enumerate(names))

    total_images = 0
    total_labels = 0
    missing_labels = []
    bad_lines = []
    class_counter = Counter()

    for split_key in ["train", "val", "test"]:
        img_dir = root / cfg[split_key]
        lab_dir = Path(str(img_dir).replace("images", "labels"))
        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        print(f"{split_key}: {len(images)} images")
        total_images += len(images)
        for img in images:
            label = lab_dir / f"{img.stem}.txt"
            if not label.exists():
                missing_labels.append(str(img))
                continue
            lines = [ln.strip() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
            total_labels += len(lines)
            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    bad_lines.append((str(label), line_no, line, "must have 5 columns"))
                    continue
                try:
                    cls = int(float(parts[0]))
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_lines.append((str(label), line_no, line, "not numeric"))
                    continue
                if cls < 0 or cls >= n_classes:
                    bad_lines.append((str(label), line_no, line, "class id out of range"))
                if any(v < 0 or v > 1 for v in vals):
                    bad_lines.append((str(label), line_no, line, "bbox values must be normalized 0..1"))
                class_counter[cls] += 1

    print(f"\nTotal images: {total_images}")
    print(f"Total boxes: {total_labels}")
    print("Class distribution:")
    for cls, count in sorted(class_counter.items()):
        print(f"  {cls} {class_names.get(cls, '?')}: {count}")

    if missing_labels:
        print(f"\n[WARN] Missing label files: {len(missing_labels)}")
        for x in missing_labels[:10]:
            print(f"  {x}")
    if bad_lines:
        print(f"\n[ERROR] Bad label lines: {len(bad_lines)}")
        for item in bad_lines[:10]:
            print(f"  {item}")
        raise SystemExit(1)
    print("\nDataset check completed.")


if __name__ == "__main__":
    main()
