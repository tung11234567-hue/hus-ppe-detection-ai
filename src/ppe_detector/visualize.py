from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .rules import Detection, PersonPPEStatus, summary_counts

GREEN = (40, 180, 40)
RED = (40, 40, 220)
YELLOW = (0, 200, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 120, 40)


def _auto_scale(
    image: np.ndarray,
    base: float = 0.65,
    min_scale: float = 0.45,
    max_scale: float = 2.6,
) -> float:
    h, w = image.shape[:2]
    ref = max(w, h)
    scale = base * (ref / 1280.0)

    return float(max(min_scale, min(max_scale, scale)))


def _auto_thickness(
    image: np.ndarray,
    base: int = 2,
    max_thickness: int = 6,
) -> int:
    h, w = image.shape[:2]
    ref = max(w, h)

    return int(max(1, min(max_thickness, round(base * ref / 1280.0))))


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = _auto_scale(image, base=0.55, min_scale=0.42, max_scale=1.8)
    thickness = _auto_thickness(image, base=2, max_thickness=5)
    pad = max(4, int(round(6 * scale)))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    h, w = image.shape[:2]

    x = max(0, min(int(x), max(0, w - tw - 2 * pad)))
    y = max(int(y), th + baseline + 2 * pad)
    y = min(y, h - baseline - pad)

    cv2.rectangle(
        image,
        (x, y - th - baseline - 2 * pad),
        (x + tw + 2 * pad, y + baseline),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + pad, y - pad),
        font,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_detection_boxes(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    out = image.copy()

    thickness = _auto_thickness(out, base=2, max_thickness=6)

    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        label = f"{d.cls_name} {d.conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, thickness)
        draw_label(out, label, x1, y1 - 6, BLUE)

    return out


def _draw_dashboard(
    out: np.ndarray,
    statuses: List[PersonPPEStatus],
) -> None:
    counts = summary_counts(statuses)

    line1 = f"Persons: {counts['persons']} | Safe: {counts['safe']} | Unsafe: {counts['unsafe']}"
    line2 = f"No helmet: {counts['no_helmet']} | No vest: {counts['no_vest']}"

    one_line = f"{line1} | {line2}"

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = _auto_scale(out, base=0.78, min_scale=0.55, max_scale=2.8)
    thickness = _auto_thickness(out, base=2, max_thickness=7)

    h, w = out.shape[:2]

    margin = max(8, int(round(14 * scale)))
    pad_x = max(8, int(round(14 * scale)))
    pad_y = max(6, int(round(10 * scale)))
    gap_y = max(4, int(round(8 * scale)))

    (tw, _), _ = cv2.getTextSize(one_line, font, scale, thickness)
    max_box_width = w - 2 * margin

    if tw + 2 * pad_x <= max_box_width:
        lines = [one_line]
    else:
        lines = [line1, line2]

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    baselines = [cv2.getTextSize(line, font, scale, thickness)[1] for line in lines]

    box_w = min(max_box_width, max(s[0] for s in sizes) + 2 * pad_x)
    line_h = max(s[1] for s in sizes)
    baseline = max(baselines)

    box_h = len(lines) * line_h + (len(lines) - 1) * gap_y + 2 * pad_y + baseline

    x1 = margin
    y1 = margin
    x2 = min(w - margin, x1 + box_w)
    y2 = min(h - margin, y1 + box_h)

    cv2.rectangle(out, (x1, y1), (x2, y2), BLACK, -1)

    y = y1 + pad_y + line_h

    for line in lines:
        cv2.putText(
            out,
            line,
            (x1 + pad_x, y),
            font,
            scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )

        y += line_h + gap_y


def draw_ppe_status(
    image: np.ndarray,
    detections: List[Detection],
    statuses: List[PersonPPEStatus],
    show_dashboard: bool = True,
) -> np.ndarray:
    out = image.copy()

    box_thin = _auto_thickness(out, base=1, max_thickness=4)
    box_normal = _auto_thickness(out, base=2, max_thickness=6)
    box_person = _auto_thickness(out, base=3, max_thickness=8)

    for d in detections:
        class_name = d.cls_name.lower().replace(" ", "_")

        if class_name not in {"person", "human", "worker", "employee"}:
            x1, y1, x2, y2 = map(int, d.xyxy)

            cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, box_thin)
            draw_label(out, f"{d.cls_name} {d.conf:.2f}", x1, y1 - 4, YELLOW)

    for status in statuses:
        p = status.person
        x1, y1, x2, y2 = map(int, p.xyxy)

        color = GREEN if not status.violations else RED
        text = "SAFE" if not status.violations else " | ".join(status.violations)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_person)
        draw_label(out, f"{text} {p.conf:.2f}", x1, y1 - 8, color)

        if status.matched_helmet is not None:
            hx1, hy1, hx2, hy2 = map(int, status.matched_helmet.xyxy)

            cv2.rectangle(out, (hx1, hy1), (hx2, hy2), GREEN, box_normal)

        if status.matched_vest is not None:
            vx1, vy1, vx2, vy2 = map(int, status.matched_vest.xyxy)

            cv2.rectangle(out, (vx1, vy1), (vx2, vy2), GREEN, box_normal)

    if show_dashboard:
        _draw_dashboard(out, statuses)

    return out