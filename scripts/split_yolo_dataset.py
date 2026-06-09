"""Split annotated YOLO images/labels into train/valid/test folders.

Expected source:
    datasets/ppe_raw/all/images/*.jpg
    datasets/ppe_raw/all/labels/*.txt

Usage:
    python scripts/split_yolo_dataset.py --src datasets/ppe_raw/all --dst datasets/ppe --require-labels
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def copy_pair(img: Path, label: Path, dst_root: Path, split: str):
    img_dst = dst_root / split / "images" / img.name
    lab_dst = dst_root / split / "labels" / f"{img.stem}.txt"
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lab_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dst)
    if label.exists():
        shutil.copy2(label, lab_dst)
    else:
        lab_dst.write_text("", encoding="utf-8")


def write_yaml(dst: Path, names: list[str]):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    lines = [
        f"path: {dst.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    Path("data/ppe.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset")
    parser.add_argument("--src", default="datasets/ppe_raw/all", help="Folder containing images/ and labels/")
    parser.add_argument("--dst", default="datasets/ppe", help="YOLO dataset output folder")
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--valid", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-labels", action="store_true", help="Skip images without .txt label")
    parser.add_argument("--names", nargs="+", default=["person", "helmet", "safety_vest"])
    parser.add_argument("--clean", action="store_true", help="Delete destination before writing")
    args = parser.parse_args()

    if abs(args.train + args.valid + args.test - 1.0) > 1e-6:
        raise SystemExit("train + valid + test must equal 1.0")

    src = Path(args.src)
    images_dir = src / "images"
    labels_dir = src / "labels"
    dst = Path(args.dst)

    images = collect_images(images_dir)
    if args.require_labels:
        images = [img for img in images if (labels_dir / f"{img.stem}.txt").exists()]

    if not images:
        raise SystemExit("No images found. Extract frames first, then annotate labels.")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    n = len(images)
    n_train = int(n * args.train)
    n_valid = int(n * args.valid)
    split_map = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test": images[n_train + n_valid:],
    }

    if args.clean and dst.exists():
        shutil.rmtree(dst)

    for split, split_images in split_map.items():
        for img in split_images:
            label = labels_dir / f"{img.stem}.txt"
            copy_pair(img, label, dst, split)
        print(f"{split}: {len(split_images)} images")

    write_yaml(dst, args.names)
    print(f"\nDataset written to: {dst.as_posix()}")
    print("YAML updated: data/ppe.yaml")


if __name__ == "__main__":
    main()
