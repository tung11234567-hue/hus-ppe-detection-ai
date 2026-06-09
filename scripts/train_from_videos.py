"""One-command helper for the video-to-training workflow.

This script extracts frames, optionally pseudo-labels them with an existing model,
splits the dataset, checks labels, then starts training.

For the FIRST training run, do not expect raw videos to be enough. You must either:
- label the extracted frames manually, or
- provide an existing PPE model through --pseudo-weights and then review labels.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="raw_videos")
    parser.add_argument("--every-sec", type=float, default=1.0)
    parser.add_argument("--pseudo-weights", default="", help="Optional existing PPE model to auto-generate draft labels")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_extract:
        run([py, "scripts/extract_frames.py", "--input", args.videos, "--output", "datasets/ppe_raw/all/images", "--every-sec", str(args.every_sec)])

    if args.pseudo_weights:
        if not Path(args.pseudo_weights).exists():
            raise SystemExit(f"pseudo weights not found: {args.pseudo_weights}")
        run([py, "scripts/pseudo_label.py", "--weights", args.pseudo_weights, "--images", "datasets/ppe_raw/all/images", "--labels", "datasets/ppe_raw/all/labels"])
        print("\nReview/fix pseudo-labels before serious training. For a quick demo, continuing is allowed.")

    run([py, "scripts/split_yolo_dataset.py", "--src", "datasets/ppe_raw/all", "--dst", "datasets/ppe", "--require-labels", "--clean"])
    run([py, "scripts/check_yolo_dataset.py", "--data", "data/ppe.yaml"])

    if not args.skip_train:
        cmd = [py, "train.py", "--data", "data/ppe.yaml", "--model", args.model, "--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--batch", str(args.batch)]
        if args.device is not None:
            cmd += ["--device", args.device]
        run(cmd)


if __name__ == "__main__":
    main()
