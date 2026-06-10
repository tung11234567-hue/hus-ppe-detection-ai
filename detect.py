from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

from src.ppe_detector.config import load_rule_config
from src.ppe_detector.io_utils import ensure_dir, save_result_json
from src.ppe_detector.model import PPEDetector
from src.ppe_detector.rules import Detection, PersonPPEStatus, analyze_ppe, summary_counts
from src.ppe_detector.visualize import draw_ppe_status

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

VIEW_WINDOW_NAME = "PPE Detection"
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect missing helmet/safety vest from image, video, or webcam")
    parser.add_argument("--weights", default="weights/best.pt", help="Trained model path")
    parser.add_argument("--source", required=True, help="Image/video path, folder, or webcam index such as 0")
    parser.add_argument("--output", default="outputs", help="Output folder")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--view", action="store_true", help="Show window while processing")
    parser.add_argument("--view-width", type=int, default=1280, help="Max preview window width")
    parser.add_argument("--view-height", type=int, default=720, help="Max preview window height")
    parser.add_argument("--no-video-fix", action="store_true", help="Disable video metadata correction")
    return parser.parse_args()


def source_to_capture(source: str) -> Union[int, str]:
    if str(source).isdigit():
        return int(source)
    return source


def create_detector(weights_path: str, conf: float = 0.05, iou: float = 0.50, device: str | None = None) -> PPEDetector:
    return PPEDetector(weights_path, conf=conf, iou=iou, device=device)


def load_rules_config_safe(config_path: str | None):
    try:
        return load_rule_config(config_path)
    except Exception:
        return load_rule_config(None)


def _parse_ratio(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() == "N/A" or ":" not in text:
        return None

    left, right = text.split(":", 1)
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return None

    if a <= 0 or b <= 0:
        return None

    return a / b


def _make_even(value: int) -> int:
    value = max(2, int(round(value)))
    return value if value % 2 == 0 else value + 1


def _read_video_metadata(source: str) -> dict[str, float]:
    if str(source).isdigit() or shutil.which("ffprobe") is None:
        return {}

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,display_aspect_ratio,sample_aspect_ratio:stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                source,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        data = json.loads(proc.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
    except Exception:
        return {}

    rotation = 0.0
    tags = stream.get("tags") or {}

    if tags.get("rotate") is not None:
        try:
            rotation = float(tags.get("rotate"))
        except ValueError:
            rotation = 0.0

    for item in stream.get("side_data_list") or []:
        if item.get("rotation") is not None:
            try:
                rotation = float(item.get("rotation"))
            except ValueError:
                pass

    w = float(stream.get("width") or 0)
    h = float(stream.get("height") or 0)

    target_aspect = _parse_ratio(stream.get("display_aspect_ratio"))
    if target_aspect is None:
        sar = _parse_ratio(stream.get("sample_aspect_ratio"))
        if sar is not None and w > 0 and h > 0:
            target_aspect = (w * sar) / h

    rotation = rotation % 360
    if math.isclose(rotation, 90, abs_tol=1) or math.isclose(rotation, 270, abs_tol=1):
        if target_aspect is not None:
            target_aspect = 1.0 / target_aspect

    info: dict[str, float] = {"rotation": rotation}
    if target_aspect is not None and target_aspect > 0:
        info["target_aspect"] = target_aspect
    return info


def _fix_video_frame(frame: np.ndarray, video_meta: dict[str, float]) -> np.ndarray:
    rotation = video_meta.get("rotation", 0) % 360

    if math.isclose(rotation, 90, abs_tol=1):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif math.isclose(rotation, 180, abs_tol=1):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif math.isclose(rotation, 270, abs_tol=1):
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    target_aspect = video_meta.get("target_aspect")
    if target_aspect:
        h, w = frame.shape[:2]
        current_aspect = w / h if h else target_aspect
        if abs(current_aspect - target_aspect) / target_aspect > 0.02:
            if target_aspect > current_aspect:
                new_w = _make_even(h * target_aspect)
                new_h = _make_even(h)
            else:
                new_w = _make_even(w)
                new_h = _make_even(w / target_aspect)
            interpolation = cv2.INTER_AREA if new_w < w or new_h < h else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    return frame


def _resize_keep_aspect(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def letterbox_to_canvas(
    image: np.ndarray,
    canvas_width: int = CANVAS_WIDTH,
    canvas_height: int = CANVAS_HEIGHT,
) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        return canvas, 1.0, 0, 0

    scale = min(canvas_width / w, canvas_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=image.dtype)
    x_offset = (canvas_width - new_w) // 2
    y_offset = (canvas_height - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas, scale, x_offset, y_offset


def _scale_box(xyxy, scale: float, x_offset: int, y_offset: int):
    x1, y1, x2, y2 = xyxy
    return (
        x1 * scale + x_offset,
        y1 * scale + y_offset,
        x2 * scale + x_offset,
        y2 * scale + y_offset,
    )


def _scale_detection(det: Detection | None, scale: float, x_offset: int, y_offset: int) -> Detection | None:
    if det is None:
        return None
    return Detection(cls_name=det.cls_name, conf=det.conf, xyxy=_scale_box(det.xyxy, scale, x_offset, y_offset))


def scale_detections(detections: list[Detection], scale: float, x_offset: int, y_offset: int) -> list[Detection]:
    scaled = []
    for det in detections:
        new_det = _scale_detection(det, scale, x_offset, y_offset)
        if new_det is not None:
            scaled.append(new_det)
    return scaled


def scale_statuses(statuses: list[PersonPPEStatus], scale: float, x_offset: int, y_offset: int) -> list[PersonPPEStatus]:
    scaled_statuses = []
    for status in statuses:
        scaled_person = _scale_detection(status.person, scale, x_offset, y_offset)
        scaled_helmet = _scale_detection(status.matched_helmet, scale, x_offset, y_offset)
        scaled_vest = _scale_detection(status.matched_vest, scale, x_offset, y_offset)
        if scaled_person is None:
            continue
        scaled_statuses.append(
            PersonPPEStatus(
                person=scaled_person,
                helmet_ok=status.helmet_ok,
                vest_ok=status.vest_ok,
                violations=list(status.violations),
                matched_helmet=scaled_helmet,
                matched_vest=scaled_vest,
            )
        )
    return scaled_statuses


def draw_fixed_dashboard(image: np.ndarray, statuses: list[PersonPPEStatus]) -> None:
    counts = summary_counts(statuses)
    text = (
        f"Persons: {counts['persons']} | "
        f"Safe: {counts['safe']} | "
        f"Unsafe: {counts['unsafe']} | "
        f"No helmet: {counts['no_helmet']} | "
        f"No vest: {counts['no_vest']}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    x = 18
    y = 38
    pad_x = 12
    pad_y = 10
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_x1 = x - pad_x
    box_y1 = y - th - pad_y
    box_x2 = x + tw + pad_x
    box_y2 = y + baseline + pad_y
    cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _init_view_window() -> None:
    try:
        cv2.namedWindow(VIEW_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    except Exception:
        cv2.namedWindow(VIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)


def detect_on_frame(
    detector: PPEDetector,
    frame_bgr: np.ndarray,
    cfg,
    use_canvas: bool = True,
    show_dashboard: bool = True,
) -> tuple[np.ndarray, list[Detection], list[PersonPPEStatus], dict[str, int]]:
    """
    Hàm AI chung cho cả detect.py và app_streamlit.py.
    Model luôn detect trên frame gốc để không giảm chất lượng.
    Nếu use_canvas=True thì scale kết quả sang canvas 1280x720 để hiển thị cố định.
    """
    detections = detector.predict_image(frame_bgr)
    statuses = analyze_ppe(detections, cfg)
    counts = summary_counts(statuses)

    if not use_canvas:
        annotated = draw_ppe_status(frame_bgr, detections, statuses)
        return annotated, detections, statuses, counts

    canvas, scale, x_offset, y_offset = letterbox_to_canvas(frame_bgr, CANVAS_WIDTH, CANVAS_HEIGHT)
    canvas_detections = scale_detections(detections, scale, x_offset, y_offset)
    canvas_statuses = scale_statuses(statuses, scale, x_offset, y_offset)

    annotated = draw_ppe_status(
        canvas,
        canvas_detections,
        canvas_statuses,
        show_dashboard=False,
    )
    if show_dashboard:
        draw_fixed_dashboard(annotated, canvas_statuses)

    return annotated, detections, statuses, counts


def process_image_bgr(
    detector: PPEDetector,
    image_bgr: np.ndarray,
    cfg,
    use_canvas: bool = True,
) -> tuple[np.ndarray, list[Detection], list[PersonPPEStatus], dict[str, int]]:
    return detect_on_frame(detector, image_bgr, cfg, use_canvas=use_canvas, show_dashboard=True)


def process_image(
    detector: PPEDetector,
    image_path: Path,
    out_dir: Path,
    config_path: str,
    save_json: bool,
) -> Path:
    cfg = load_rules_config_safe(config_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    annotated, detections, statuses, _counts = process_image_bgr(detector, image, cfg, use_canvas=True)

    out_path = out_dir / f"{image_path.stem}_ppe.jpg"
    cv2.imwrite(str(out_path), annotated)

    if save_json:
        save_result_json(out_dir / f"{image_path.stem}_ppe.json", detections, statuses)

    return out_path


def process_video_to_path(
    detector: PPEDetector,
    source: str,
    out_path: str | Path,
    config_path: str,
    max_seconds: int = 0,
    frame_skip: int = 1,
    fix_video: bool = True,
    progress_callback=None,
    preview_callback=None,
) -> dict[str, Any]:
    cfg = load_rules_config_safe(config_path)
    cap = cv2.VideoCapture(source_to_capture(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 1 or fps > 240:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames
    if max_seconds > 0:
        max_frames = min(max_frames, int(max_seconds * fps)) if total_frames else int(max_seconds * fps)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, frame_skip),
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {out_path}")

    video_meta = _read_video_metadata(source) if fix_video else {}

    frame_idx = 0
    processed = 0
    last_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}
    max_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}
    sum_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}

    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame_idx += 1
            if frame_idx % max(1, frame_skip) != 0:
                continue

            frame = _fix_video_frame(frame, video_meta) if fix_video else frame
            annotated, detections, statuses, counts = detect_on_frame(detector, frame, cfg, use_canvas=True, show_dashboard=True)

            writer.write(annotated)
            processed += 1
            last_counts = counts

            for k in max_counts:
                max_counts[k] = max(max_counts[k], counts.get(k, 0))
                sum_counts[k] += counts.get(k, 0)

            if preview_callback is not None and processed % 10 == 0:
                preview_callback(annotated)

            if progress_callback is not None:
                progress = min(frame_idx / max_frames, 1.0) if max_frames else 0.0
                progress_callback(progress, processed, frame_idx, max_frames or total_frames, counts)

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - start
    avg_counts = {k: round(v / processed, 2) if processed else 0 for k, v in sum_counts.items()}

    return {
        "output_path": str(out_path),
        "processed_frames": processed,
        "elapsed": elapsed,
        "last_counts": last_counts,
        "max_counts": max_counts,
        "avg_counts": avg_counts,
    }


def process_video(
    detector: PPEDetector,
    source: str,
    out_dir: Path,
    config_path: str,
    view: bool,
    view_width: int = 1280,
    view_height: int = 720,
    fix_video: bool = True,
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ppe_video_{timestamp}.mp4"

    preview_frame: np.ndarray | None = None

    def preview_callback(frame: np.ndarray) -> None:
        nonlocal preview_frame
        preview_frame = frame

    if view:
        _init_view_window()

    # Tự xử lý vòng lặp để hiển thị realtime bằng OpenCV.
    cfg = load_rules_config_safe(config_path)
    cap = cv2.VideoCapture(source_to_capture(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 1 or fps > 240:
        fps = 25

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_WIDTH, CANVAS_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {out_path}")

    video_meta = _read_video_metadata(source) if fix_video else {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _fix_video_frame(frame, video_meta) if fix_video else frame
            annotated, _detections, _statuses, _counts = detect_on_frame(detector, frame, cfg, use_canvas=True, show_dashboard=True)
            writer.write(annotated)

            if view:
                preview = _resize_keep_aspect(annotated, view_width, view_height)
                ph, pw = preview.shape[:2]
                try:
                    if cv2.getWindowProperty(VIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    cv2.resizeWindow(VIEW_WINDOW_NAME, pw, ph)
                    cv2.imshow(VIEW_WINDOW_NAME, preview)
                except cv2.error:
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        writer.release()
        if view:
            cv2.destroyAllWindows()

    return out_path


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output)
    detector = create_detector(args.weights, conf=args.conf, iou=args.iou, device=args.device)
    source = Path(args.source)

    if source.exists() and source.is_dir():
        for image_path in source.iterdir():
            if image_path.suffix.lower() in IMAGE_EXTS:
                out = process_image(detector, image_path, out_dir, args.config, args.save_json)
                print(f"Saved: {out}")

    elif source.exists() and source.suffix.lower() in IMAGE_EXTS:
        out = process_image(detector, source, out_dir, args.config, args.save_json)
        print(f"Saved: {out}")

    else:
        out = process_video(
            detector,
            args.source,
            out_dir,
            args.config,
            args.view,
            view_width=args.view_width,
            view_height=args.view_height,
            fix_video=not args.no_video_fix,
        )
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
