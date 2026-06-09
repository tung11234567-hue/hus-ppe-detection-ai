from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.ppe_detector.config import load_rule_config
from src.ppe_detector.model import PPEDetector
from src.ppe_detector.rules import analyze_ppe, summary_counts
from src.ppe_detector.visualize import draw_ppe_status


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="PPE HUS AI Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #0f766e 100%);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 18px 45px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 12px 28px rgba(0,0,0,0.16);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .safe {
        color: #22c55e;
    }

    .unsafe {
        color: #ef4444;
    }

    .warn {
        color: #f59e0b;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        padding: 0.8rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    section[data-testid="stSidebar"] {
        background: #020617;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# UTILS
# =========================

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


@st.cache_resource(show_spinner=False)
def load_detector_cached(weights_path: str, device_value: str | None) -> PPEDetector:
    return PPEDetector(weights_path, conf=0.25, iou=0.50, device=device_value)


@st.cache_data(show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def safe_load_cfg(config_path: str):
    try:
        return load_rule_config(config_path)
    except Exception:
        st.warning("Không đọc được config rules. App dùng rule mặc định.")
        return load_rule_config(None)


def get_detector(weights_path: str, conf: float, iou: float, device: str | None) -> PPEDetector:
    detector = load_detector_cached(weights_path, device)
    detector.conf = conf
    detector.iou = iou
    detector.device = device
    return detector


def detections_to_df(detections) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(detections, start=1):
        x1, y1, x2, y2 = d.xyxy
        rows.append(
            {
                "#": i,
                "class": d.cls_name,
                "conf": round(float(d.conf), 3),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return pd.DataFrame(rows)


def statuses_to_df(statuses) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(statuses, start=1):
        x1, y1, x2, y2 = s.person.xyxy
        rows.append(
            {
                "person": i,
                "status": "SAFE" if not s.violations else " + ".join(s.violations),
                "helmet_ok": bool(s.helmet_ok),
                "vest_ok": bool(s.vest_ok),
                "person_conf": round(float(s.person.conf), 3),
                "box": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}",
            }
        )
    return pd.DataFrame(rows)


def build_json_result(detections, statuses, counts: dict[str, int]) -> str:
    data: dict[str, Any] = {
        "summary": counts,
        "detections": [
            {
                "class": d.cls_name,
                "conf": float(d.conf),
                "xyxy": [float(v) for v in d.xyxy],
            }
            for d in detections
        ],
        "persons": [
            {
                "person_conf": float(s.person.conf),
                "person_xyxy": [float(v) for v in s.person.xyxy],
                "helmet_ok": bool(s.helmet_ok),
                "vest_ok": bool(s.vest_ok),
                "violations": list(s.violations),
                "status": "SAFE" if not s.violations else " + ".join(s.violations),
            }
            for s in statuses
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def encode_png(image_rgb: np.ndarray) -> bytes:
    image_bgr = rgb_to_bgr(image_rgb)
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Không encode được ảnh PNG.")
    return buf.tobytes()


def run_detection_on_bgr(
    image_bgr: np.ndarray,
    weights_path: str,
    config_path: str,
    conf: float,
    iou: float,
    device: str | None,
):
    detector = get_detector(weights_path, conf, iou, device)
    cfg = safe_load_cfg(config_path)

    detections = detector.predict_image(image_bgr)
    statuses = analyze_ppe(detections, cfg)
    counts = summary_counts(statuses)
    annotated_bgr = draw_ppe_status(image_bgr, detections, statuses)

    return annotated_bgr, detections, statuses, counts


def show_summary_cards(counts: dict[str, int]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Persons", counts.get("persons", 0))
    with c2:
        st.metric("Safe", counts.get("safe", 0))
    with c3:
        st.metric("Unsafe", counts.get("unsafe", 0))
    with c4:
        st.metric("No helmet", counts.get("no_helmet", 0))
    with c5:
        st.metric("No vest", counts.get("no_vest", 0))


def show_result_tables(detections, statuses, counts: dict[str, int]) -> None:
    tab1, tab2, tab3 = st.tabs(["Trạng thái từng người", "Tất cả detection", "JSON"])

    with tab1:
        df_status = statuses_to_df(statuses)
        if df_status.empty:
            st.info("Không phát hiện được person hợp lệ trong ảnh/frame này.")
        else:
            st.dataframe(df_status, use_container_width=True, hide_index=True)

    with tab2:
        df_det = detections_to_df(detections)
        if df_det.empty:
            st.info("Không có detection nào vượt ngưỡng conf hiện tại.")
        else:
            st.dataframe(df_det, use_container_width=True, hide_index=True)

    with tab3:
        st.code(build_json_result(detections, statuses, counts), language="json")


def process_uploaded_image(uploaded_file, weights_path: str, config_path: str, conf: float, iou: float, device: str | None) -> None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = rgb_to_bgr(image_rgb)

    with st.spinner("Đang chạy YOLO trên ảnh..."):
        start = time.time()
        annotated_bgr, detections, statuses, counts = run_detection_on_bgr(
            image_bgr, weights_path, config_path, conf, iou, device
        )
        elapsed = time.time() - start

    annotated_rgb = bgr_to_rgb(annotated_bgr)

    show_summary_cards(counts)
    st.caption(f"Thời gian xử lý: {elapsed:.2f}s | Detections: {len(detections)}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Kết quả AI")
        st.image(annotated_rgb, use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Tải ảnh kết quả PNG",
            data=encode_png(annotated_rgb),
            file_name="ppe_result.png",
            mime="image/png",
            use_container_width=True,
        )
    with b2:
        st.download_button(
            "Tải JSON kết quả",
            data=build_json_result(detections, statuses, counts),
            file_name="ppe_result.json",
            mime="application/json",
            use_container_width=True,
        )

    show_result_tables(detections, statuses, counts)


def process_video_file(
    uploaded_file,
    weights_path: str,
    config_path: str,
    conf: float,
    iou: float,
    device: str | None,
    max_seconds: int,
    frame_skip: int,
) -> None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_file.getbuffer())
        input_path = tmp_in.name

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Không mở được video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 1 or fps > 240:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        st.error("Video lỗi kích thước frame.")
        cap.release()
        return

    max_frames = total_frames
    if max_seconds > 0:
        max_frames = min(max_frames, int(max_seconds * fps)) if total_frames else int(max_seconds * fps)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(1, frame_skip),
        (w, h),
    )

    if not writer.isOpened():
        st.error("Không tạo được file video output.")
        cap.release()
        return

    progress = st.progress(0)
    status_box = st.empty()
    preview_box = st.empty()

    detector = get_detector(weights_path, conf, iou, device)
    cfg = safe_load_cfg(config_path)

    frame_idx = 0
    processed = 0
    last_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}
    max_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}
    sum_counts = {"persons": 0, "safe": 0, "unsafe": 0, "no_helmet": 0, "no_vest": 0}

    start = time.time()

    with st.spinner("Đang xử lý video..."):
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame_idx += 1

            if frame_idx % max(1, frame_skip) != 0:
                continue

            detections = detector.predict_image(frame)
            statuses = analyze_ppe(detections, cfg)
            counts = summary_counts(statuses)
            annotated = draw_ppe_status(frame, detections, statuses)

            writer.write(annotated)

            processed += 1
            last_counts = counts

            for k in max_counts:
                max_counts[k] = max(max_counts[k], counts.get(k, 0))
                sum_counts[k] += counts.get(k, 0)

            if processed % 10 == 0:
                preview_box.image(bgr_to_rgb(annotated), caption="Preview frame đang xử lý", use_container_width=True)

            if max_frames:
                progress.progress(min(frame_idx / max_frames, 1.0))

            status_box.write(
                f"Đã xử lý {processed} frame | Frame gốc {frame_idx}/{max_frames if max_frames else total_frames} | "
                f"Persons: {counts.get('persons', 0)} | Unsafe: {counts.get('unsafe', 0)}"
            )

    cap.release()
    writer.release()

    elapsed = time.time() - start

    if processed == 0:
        st.warning("Không xử lý được frame nào.")
        return

    avg_counts = {k: round(v / processed, 2) for k, v in sum_counts.items()}

    st.success(f"Xử lý xong video trong {elapsed:.1f}s. Processed frames: {processed}")

    st.subheader("Thống kê video")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("Frame cuối")
        show_summary_cards(last_counts)

    with c2:
        st.write("Max trong video")
        st.json(max_counts)

    with c3:
        st.write("Trung bình/frame")
        st.json(avg_counts)

    st.subheader("Video kết quả")
    video_bytes = read_file_bytes(output_path)
    st.video(video_bytes)

    st.download_button(
        "Tải video kết quả",
        data=video_bytes,
        file_name="ppe_video_result.mp4",
        mime="video/mp4",
        use_container_width=True,
    )


# =========================
# HEADER
# =========================

st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">PPE HUS AI Detection</div>
    <p class="hero-subtitle">
        Demo AI phát hiện người không đội mũ bảo hộ hoặc không mặc áo phản quang trong ảnh, video và webcam.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("Cấu hình model")

    weights = st.text_input("Weights", value="weights/best.pt")
    config_path = st.text_input("Config rules", value="configs/default.yaml")

    conf = st.slider(
        "Confidence",
        min_value=0.01,
        max_value=0.95,
        value=0.05,
        step=0.01,
        help="Conf thấp thì ít bỏ sót hơn nhưng dễ nhận nhầm. Conf cao thì ít nhận nhầm hơn nhưng dễ bỏ sót.",
    )

    iou = st.slider(
        "IoU",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="IoU dùng để lọc box trùng nhau. Thường để 0.45–0.50.",
    )

    device_text = st.text_input("Device", value="", help="Để trống = auto. Có thể nhập cpu hoặc 0 nếu có GPU.")
    device = device_text.strip() or None

    st.divider()

    st.header("Thông tin nhanh")
    st.write("**Class YOLO:**")
    st.caption("person, helmet, safety_vest, no_helmet, no_vest")

    st.write("**Trạng thái suy luận:**")
    st.caption("SAFE = có cả helmet + safety_vest")
    st.caption("NO_HELMET = thiếu mũ")
    st.caption("NO_VEST = thiếu áo phản quang")

    st.divider()

    st.header("Video")
    max_seconds = st.slider("Giới hạn giây xử lý", 0, 300, 60, 10, help="0 = xử lý hết video.")
    frame_skip = st.slider("Frame skip", 1, 10, 1, 1, help="1 = xử lý mọi frame. 2 = cách 1 frame xử lý 1 frame.")


# =========================
# VALIDATE
# =========================

if not Path(weights).exists():
    st.warning(f"Chưa thấy file weights: `{weights}`. Nếu đang chạy trên máy khác, kiểm tra lại đường dẫn.")
if not Path(config_path).exists():
    st.info(f"Không thấy config `{config_path}`. App sẽ dùng rule mặc định nếu chạy detection.")


# =========================
# MAIN TABS
# =========================

tab_img, tab_video, tab_cam, tab_help = st.tabs(
    ["Ảnh", "Video", "Webcam", "Giải thích"]
)


with tab_img:
    st.subheader("Nhận diện trên ảnh")

    uploaded_img = st.file_uploader(
        "Upload ảnh công trường",
        type=IMAGE_TYPES,
        key="image_uploader",
    )

    if uploaded_img is None:
        st.info("Upload ảnh `.jpg`, `.png`, `.webp` để chạy demo.")
    else:
        try:
            process_uploaded_image(uploaded_img, weights, config_path, conf, iou, device)
        except Exception as exc:
            st.error(f"Lỗi khi xử lý ảnh: {exc}")


with tab_video:
    st.subheader("Nhận diện trên video")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=VIDEO_TYPES,
        key="video_uploader",
    )

    if uploaded_video is None:
        st.info("Upload video `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` để chạy demo.")
    else:
        st.caption("Video dài có thể xử lý lâu. Có thể giảm số giây xử lý hoặc tăng frame skip trong sidebar.")
        if st.button("Chạy nhận diện video", type="primary", use_container_width=True):
            try:
                process_video_file(
                    uploaded_video,
                    weights,
                    config_path,
                    conf,
                    iou,
                    device,
                    max_seconds=max_seconds,
                    frame_skip=frame_skip,
                )
            except Exception as exc:
                st.error(f"Lỗi khi xử lý video: {exc}")


with tab_cam:
    st.subheader("Chụp ảnh từ webcam")

    captured = st.camera_input("Chụp một ảnh từ webcam")

    if captured is not None:
        try:
            process_uploaded_image(captured, weights, config_path, conf, iou, device)
        except Exception as exc:
            st.error(f"Lỗi khi xử lý ảnh webcam: {exc}")


with tab_help:
    st.subheader("Cách hiểu kết quả")

    st.markdown(
        """
### 1. Class và trạng thái khác nhau

**Class YOLO detect trực tiếp:**

- `person`: người/công nhân.
- `helmet`: mũ bảo hộ.
- `safety_vest`: áo phản quang.
- `no_helmet`: vùng/người thiếu mũ.
- `no_vest`: vùng/người thiếu áo phản quang.

**Trạng thái app suy luận:**

- `SAFE`: người có đủ mũ bảo hộ và áo phản quang.
- `NO_HELMET`: người thiếu mũ.
- `NO_VEST`: người thiếu áo phản quang.

`SAFE` không phải class train trực tiếp. `SAFE` là kết quả sau khi code kiểm tra người đó có đủ `helmet` và `safety_vest`.

---

### 2. Confidence là gì?

`conf` là độ tự tin tối thiểu để giữ lại một detection.

- Conf thấp: ít bỏ sót hơn, nhưng dễ nhận nhầm.
- Conf cao: ít nhận nhầm hơn, nhưng dễ bỏ sót.

Demo PPE nên thử từ `0.05` đến `0.25`.

---

### 3. IoU là gì?

`IoU` dùng để lọc các box trùng nhau.

- IoU thấp: lọc box trùng mạnh hơn.
- IoU cao: giữ lại nhiều box hơn.

Thường để `0.45–0.50`.

---

### 4. Code hiện tại có tracking chưa?

Chưa. App hiện tại là detection + counting theo từng ảnh/frame.
Muốn có tracking ID cần dùng `model.track(..., persist=True, tracker="bytetrack.yaml")`.
"""
    )