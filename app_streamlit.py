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

# App Streamlit chỉ làm giao diện.
# Toàn bộ logic detect/canvas/video dùng chung nằm trong detect.py.
from detect import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    create_detector,
    load_rules_config_safe,
    process_image_bgr,
    process_video_to_path,
)


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
# UI UTILS
# =========================

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


@st.cache_resource(show_spinner=False)
def load_detector_cached(weights_path: str, conf: float, iou: float, device_value: str | None):
    return create_detector(weights_path, conf=conf, iou=iou, device=device_value)


@st.cache_data(show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def get_detector(weights_path: str, conf: float, iou: float, device: str | None):
    detector = load_detector_cached(weights_path, conf, iou, device)
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
                "box_original": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}",
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

    detector = get_detector(weights_path, conf, iou, device)
    cfg = load_rules_config_safe(config_path)

    with st.spinner("Đang chạy YOLO trên ảnh..."):
        start = time.time()
        annotated_bgr, detections, statuses, counts = process_image_bgr(
            detector,
            image_bgr,
            cfg,
            use_canvas=True,
        )
        elapsed = time.time() - start

    annotated_rgb = bgr_to_rgb(annotated_bgr)

    show_summary_cards(counts)
    st.caption(
        f"Thời gian xử lý: {elapsed:.2f}s | Detections: {len(detections)} | "
        f"Khung kết quả: {CANVAS_WIDTH}x{CANVAS_HEIGHT}"
    )

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

    detector = get_detector(weights_path, conf, iou, device)

    progress = st.progress(0)
    status_box = st.empty()
    preview_box = st.empty()

    def progress_callback(progress_value, processed, frame_idx, total, counts):
        progress.progress(progress_value)
        status_box.write(
            f"Đã xử lý {processed} frame | Frame gốc {frame_idx}/{total} | "
            f"Persons: {counts.get('persons', 0)} | Unsafe: {counts.get('unsafe', 0)}"
        )

    def preview_callback(frame_bgr):
        preview_box.image(bgr_to_rgb(frame_bgr), caption="Preview frame đang xử lý", use_container_width=True)

    with st.spinner("Đang xử lý video..."):
        result = process_video_to_path(
            detector=detector,
            source=input_path,
            out_path=output_path,
            config_path=config_path,
            max_seconds=max_seconds,
            frame_skip=frame_skip,
            fix_video=True,
            progress_callback=progress_callback,
            preview_callback=preview_callback,
        )

    if result["processed_frames"] == 0:
        st.warning("Không xử lý được frame nào.")
        return

    st.success(f"Xử lý xong video trong {result['elapsed']:.1f}s. Processed frames: {result['processed_frames']}")

    st.subheader("Thống kê video")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Frame cuối")
        show_summary_cards(result["last_counts"])
    with c2:
        st.write("Max trong video")
        st.json(result["max_counts"])
    with c3:
        st.write("Trung bình/frame")
        st.json(result["avg_counts"])

    st.subheader(f"Video kết quả ({CANVAS_WIDTH}x{CANVAS_HEIGHT})")
    video_bytes = read_file_bytes(result["output_path"])
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

tab_img, tab_video, tab_cam, tab_help = st.tabs(["Ảnh", "Video", "Webcam", "Giải thích"])

with tab_img:
    st.subheader("Nhận diện trên ảnh")
    uploaded_img = st.file_uploader("Upload ảnh công trường", type=IMAGE_TYPES, key="image_uploader")

    if uploaded_img is None:
        st.info("Upload ảnh `.jpg`, `.png`, `.webp` để chạy demo.")
    else:
        try:
            process_uploaded_image(uploaded_img, weights, config_path, conf, iou, device)
        except Exception as exc:
            st.error(f"Lỗi khi xử lý ảnh: {exc}")

with tab_video:
    st.subheader("Nhận diện trên video")
    uploaded_video = st.file_uploader("Upload video", type=VIDEO_TYPES, key="video_uploader")

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
        f"""
### 1. File nào xử lý AI?

App Streamlit hiện tại chỉ làm giao diện: upload ảnh/video, hiển thị kết quả, bảng thống kê và nút tải file.

Toàn bộ xử lý AI chính nằm trong `detect.py`, gồm:

- Load YOLO model
- Detect ảnh/frame
- Suy luận SAFE / NO_HELMET / NO_VEST
- Vẽ bounding box
- Ép kết quả vào khung cố định `{CANVAS_WIDTH}x{CANVAS_HEIGHT}`
- Xử lý video output

Vì vậy sửa logic detect thì sửa trong `detect.py`.

---

### 2. Class và trạng thái khác nhau

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

### 3. Confidence là gì?

`conf` là độ tự tin tối thiểu để giữ lại một detection.

- Conf thấp: ít bỏ sót hơn, nhưng dễ nhận nhầm.
- Conf cao: ít nhận nhầm hơn, nhưng dễ bỏ sót.

Demo PPE nên thử từ `0.05` đến `0.25`.

---

### 4. IoU là gì?

`IoU` dùng để lọc các box trùng nhau.

- IoU thấp: lọc box trùng mạnh hơn.
- IoU cao: giữ lại nhiều box hơn.

Thường để `0.45–0.50`.

---

### 5. Code hiện tại có tracking thật chưa?

Không dùng tracking thật để tránh làm giảm/mất detection.

Bản này ưu tiên nhận diện PPE ổn định. Nếu cần tracking thật, nên thêm sau và kiểm thử riêng.
"""
    )
