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

import cv2
import numpy as np


def _text_style(image):
    h, w = image.shape[:2]

    # Font tự co theo kích thước ảnh, tránh lúc to lúc bé
    scale = max(0.35, min(0.55, w / 2200))
    thickness = 1 if w < 1400 else 2

    return scale, thickness


def _draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = _text_style(image)

    x = max(0, int(x))
    y = max(18, int(y))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Nếu label quá dài thì cắt ngắn
    max_width = 180
    if tw > max_width:
        text = text[:18] + "..."
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(
        image,
        (x, y - th - 5),
        (x + tw + 6, y + baseline + 2),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 3, y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _status_text(status):
    if not status.violations:
        return "SAFE"
    return "+".join(status.violations)


def draw_ppe_status(image, detections, statuses, show_dashboard=True):
    """
    Vẽ kết quả PPE gọn hơn:
    - Person box: xanh nếu SAFE, đỏ nếu vi phạm
    - Helmet/vest box: vàng, label nhỏ
    - Không vẽ label quá to che ảnh
    """

    annotated = image.copy()

    # Vẽ helmet / vest / no_helmet / no_vest trước
    for det in detections:
        name = str(det.cls_name)
        norm = name.lower().replace(" ", "_")

        if norm in ["person", "worker", "employee"]:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy)

        if norm in ["helmet", "hardhat", "hard_hat", "safety_helmet"]:
            color = (0, 220, 255)      # vàng
        elif norm in ["vest", "safety_vest", "reflective_vest", "high_vis_vest"]:
            color = (0, 200, 255)      # vàng cam
        elif norm in ["no_helmet", "no-helmet", "no_hardhat", "no-hardhat"]:
            color = (0, 0, 255)        # đỏ
        elif norm in ["no_vest", "no-vest", "no_safety_vest", "no-safety-vest"]:
            color = (0, 80, 255)       # cam đỏ
        else:
            color = (180, 180, 180)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        label = f"{name} {float(det.conf):.2f}"
        _draw_label(annotated, label, x1, y1 - 4, color)

    # Vẽ person + trạng thái
    for idx, status in enumerate(statuses, start=1):
        x1, y1, x2, y2 = map(int, status.person.xyxy)

        safe = not status.violations
        color = (0, 180, 0) if safe else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        person_id = getattr(status, "person_id", None)
        if person_id is not None:
            label = f"ID {person_id}"
        else:
            label = _status_text(status)

        _draw_label(annotated, label, x1, y1 - 6, color)

    # Dashboard nhỏ góc trên trái
    if show_dashboard:
        persons = len(statuses)
        unsafe = sum(bool(s.violations) for s in statuses)
        safe = persons - unsafe
        no_helmet = sum("NO_HELMET" in s.violations for s in statuses)
        no_vest = sum("NO_VEST" in s.violations for s in statuses)

        text1 = f"Persons: {persons} | Safe: {safe} | Unsafe: {unsafe}"
        text2 = f"No helmet: {no_helmet} | No vest: {no_vest}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = _text_style(annotated)
        scale = max(0.42, min(scale, 0.55))

        x, y = 12, 25

        (tw1, th1), _ = cv2.getTextSize(text1, font, scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(text2, font, scale, thickness)
        box_w = max(tw1, tw2) + 18
        box_h = th1 + th2 + 24

        cv2.rectangle(annotated, (6, 6), (6 + box_w, 6 + box_h), (0, 0, 0), -1)

        cv2.putText(annotated, text1, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(annotated, text2, (x, y + th1 + 12), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated