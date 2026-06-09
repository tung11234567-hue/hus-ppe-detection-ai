"""Extract training frames from raw construction-site videos.

Usage:
    python scripts/extract_frames.py --input raw_videos --output datasets/ppe_raw/all/images --every-sec 1

This script does NOT create labels. You still need to annotate the extracted images
with LabelImg, CVAT, Roboflow, or run scripts/pseudo_label.py with an existing model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def iter_videos(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p


def safe_stem(path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    short_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{short_hash}"


def blur_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max(frame, max_size: int):
    if max_size <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def frame_difference(frame_a, frame_b) -> float:
    if frame_a is None or frame_b is None:
        return 999.0
    a = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), (64, 64))
    b = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), (64, 64))
    return float(cv2.absdiff(a, b).mean())


def extract_video(video_path: Path, output_dir: Path, every_sec: float, frame_interval: int | None,
                  max_size: int, min_blur: float, min_diff: float, jpeg_quality: int,
                  rows: list[dict]) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = frame_interval if frame_interval and frame_interval > 0 else max(1, int(round(fps * every_sec)))
    prefix = safe_stem(video_path)

    saved = 0
    idx = 0
    last_saved_small = None
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        score = blur_score(frame)
        diff = frame_difference(frame, last_saved_small)
        if score < min_blur or diff < min_diff:
            idx += 1
            continue

        frame_out = resize_max(frame, max_size)
        timestamp = idx / fps
        name = f"{prefix}_f{idx:07d}_t{timestamp:08.2f}.jpg"
        out_path = output_dir / name
        cv2.imwrite(str(out_path), frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        last_saved_small = frame.copy()
        saved += 1
        rows.append({
            "image": str(out_path.as_posix()),
            "source_video": str(video_path.as_posix()),
            "frame_index": idx,
            "timestamp_sec": f"{timestamp:.2f}",
            "fps": f"{fps:.2f}",
            "blur_score": f"{score:.2f}",
            "diff_score": f"{diff:.2f}",
            "video_total_frames": total,
        })
        idx += 1

    cap.release()
    print(f"[OK] {video_path.name}: saved {saved} frames")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract useful frames from videos for YOLO training")
    parser.add_argument("--input", default="raw_videos", help="Video file or folder containing videos")
    parser.add_argument("--output", default="datasets/ppe_raw/all/images", help="Output image folder")
    parser.add_argument("--every-sec", type=float, default=1.0, help="Sample 1 frame every N seconds")
    parser.add_argument("--frame-interval", type=int, default=0, help="Alternative: sample every N frames")
    parser.add_argument("--max-size", type=int, default=1280, help="Resize so longest side <= this; 0 disables resize")
    parser.add_argument("--min-blur", type=float, default=40.0, help="Skip blurry frames below this Laplacian variance")
    parser.add_argument("--min-diff", type=float, default=2.0, help="Skip near-duplicate frames below this mean difference")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--metadata", default="datasets/ppe_raw/frame_metadata.csv")
    args = parser.parse_args()

    videos = list(iter_videos(Path(args.input)))
    if not videos:
        raise SystemExit(f"No video found in {args.input}")

    rows: list[dict] = []
    total = 0
    for video in videos:
        total += extract_video(
            video_path=video,
            output_dir=Path(args.output),
            every_sec=args.every_sec,
            frame_interval=args.frame_interval if args.frame_interval > 0 else None,
            max_size=args.max_size,
            min_blur=args.min_blur,
            min_diff=args.min_diff,
            jpeg_quality=args.jpeg_quality,
            rows=rows,
        )

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nDone. Total saved frames: {total}")
    print(f"Images: {Path(args.output).as_posix()}")
    print(f"Metadata: {metadata_path.as_posix()}")


if __name__ == "__main__":
    main()
