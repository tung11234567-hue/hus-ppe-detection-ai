# HUS PPE Detection AI

Project AI phát hiện vi phạm bảo hộ lao động tại công trường bằng YOLO26x.

## Chức năng

- Phát hiện person, helmet, safety_vest, no_helmet, no_vest
- Suy luận trạng thái SAFE / NO_HELMET / NO_VEST
- Hỗ trợ ảnh, video, webcam qua Streamlit
- Có tracking ID cho person trong video

## Cài đặt

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Model weights

Không upload best.pt lên GitHub vì file nặng.

Tải best.pt riêng rồi đặt vào:

weights/best.pt
Chạy web demo
streamlit run app_streamlit.py
Test ảnh
python detect.py --weights weights\best.pt --source "test_media\test1.jpg" --view --conf 0.05
Test video
python detect.py --weights weights\best.pt --source "raw_videos\test2.mp4" --view --conf 0.05
Thông tin model
Model train: YOLO26x
Framework: Ultralytics
Dataset format: YOLOv8 / Ultralytics YOLO
Dataset gốc: 16 class, convert về 5 class
Best mAP50 hiện tại: 0.725
"@ | Set-Content -Path README.md -Encoding UTF8