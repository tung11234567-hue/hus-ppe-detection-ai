# Hướng dẫn train từ video công trường

## 1. Sự thật quan trọng

Không thể chỉ ném video thô vào rồi model tự biết đâu là `person`, `helmet`, `safety_vest`.
Train object detection cần cặp dữ liệu:

```text
ảnh frame + file nhãn bounding box
```

Vì vậy workflow đúng là:

```text
video trên mạng / video tự quay
→ cắt frame
→ gán nhãn person, helmet, safety_vest
→ chia train/valid/test
→ train YOLO
→ dùng model detect video/webcam
```

Có thể tự động phần cắt frame, chia dataset, train. Phần gán nhãn chỉ có thể bán tự động, cần người kiểm tra lại.

## 2. Chuẩn bị video

Tạo thư mục:

```bash
mkdir raw_videos
```

Bỏ video `.mp4`, `.avi`, `.mov`, `.mkv` vào đó:

```text
raw_videos/
├── site_01.mp4
├── site_02.mp4
└── site_03.mp4
```

Nên dùng video có cảnh công trường, người đi lại, đủ sáng, nhiều góc camera. Không nên lấy quá nhiều frame gần giống nhau.

## 3. Cắt frame tự động

Cắt 1 ảnh mỗi giây:

```bash
python scripts/extract_frames.py --input raw_videos --output datasets/ppe_raw/all/images --every-sec 1
```

Nếu video dài, có thể cắt thưa hơn:

```bash
python scripts/extract_frames.py --input raw_videos --output datasets/ppe_raw/all/images --every-sec 2
```

Kết quả:

```text
datasets/ppe_raw/all/images/*.jpg
datasets/ppe_raw/frame_metadata.csv
```

## 4. Gán nhãn

Class chuẩn của project:

```text
0 person
1 helmet
2 safety_vest
```

Mỗi ảnh cần khoanh:

- toàn thân người: `person`
- mũ bảo hộ: `helmet`
- áo phản quang: `safety_vest`

Có thể dùng 1 trong các tool:

- LabelImg: nhẹ, chạy local, xuất YOLO txt.
- CVAT: mạnh hơn, phù hợp nhiều ảnh.
- Roboflow: dễ dùng web, có thể export YOLOv8/YOLO11.

Với LabelImg:

```bash
pip install labelImg
labelImg datasets/ppe_raw/all/images data/classes.txt
```

Trong LabelImg nhớ chọn format YOLO, lưu label vào:

```text
datasets/ppe_raw/all/labels/
```

Sau khi gán nhãn, mỗi ảnh sẽ có file `.txt` cùng tên:

```text
datasets/ppe_raw/all/images/site_01_xxx.jpg
datasets/ppe_raw/all/labels/site_01_xxx.txt
```

Nội dung file nhãn YOLO có dạng:

```text
class_id x_center y_center width height
```

Tất cả tọa độ đã được chuẩn hóa từ 0 đến 1.

## 5. Chia dataset train/valid/test

```bash
python scripts/split_yolo_dataset.py --src datasets/ppe_raw/all --dst datasets/ppe --require-labels --clean
```

Project sẽ tự tạo:

```text
datasets/ppe/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
└── test/labels
```

Đồng thời cập nhật `data/ppe.yaml`.

## 6. Check dataset trước khi train

```bash
python scripts/check_yolo_dataset.py --data data/ppe.yaml
```

Nếu báo lỗi class id, bbox, thiếu label thì sửa trước khi train.

## 7. Train

Máy yếu hoặc không có GPU:

```bash
python train.py --data data/ppe.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8
```

Có GPU:

```bash
python train.py --data data/ppe.yaml --model yolo11s.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

Model tốt nhất nằm ở:

```text
runs/train/ppe_yolo/weights/best.pt
```

Copy sang:

```text
weights/best.pt
```

## 8. Detect lại video

```bash
python detect.py --weights weights/best.pt --source raw_videos/site_01.mp4 --view
```

Hoặc chạy web demo:

```bash
streamlit run app_streamlit.py
```

## 9. Bán tự động gán nhãn bằng pseudo-label

Sau khi đã có một model PPE ban đầu, có thể dùng nó để tự tạo nhãn nháp cho video mới:

```bash
python scripts/pseudo_label.py --weights weights/best.pt --images datasets/ppe_raw/all/images --labels datasets/ppe_raw/all/labels --conf 0.45
```

Sau đó vẫn nên mở bằng LabelImg/CVAT để sửa nhãn sai. Không nên train luôn từ pseudo-label chưa kiểm tra, vì model sẽ học theo lỗi cũ.

## 10. Một lệnh gần tự động

Sau khi đã có label hoặc có pseudo weights:

```bash
python scripts/train_from_videos.py --videos raw_videos --every-sec 1 --epochs 80
```

Nếu dùng pseudo-label:

```bash
python scripts/train_from_videos.py --videos raw_videos --every-sec 1 --pseudo-weights weights/best.pt --epochs 80
```

## 11. Gợi ý số lượng dữ liệu cho bài cuối kỳ

Mức demo ổn:

- 10-20 video ngắn.
- Cắt 500-1500 frame.
- Gán nhãn kỹ khoảng 300-800 ảnh.
- Valid/test nên có cảnh khác với train.

Mức tốt hơn:

- 1500-3000 ảnh có nhãn.
- Đủ trường hợp: có mũ, không mũ, có áo, không áo, bị che khuất, ánh sáng yếu, xa/gần camera.

## 12. Lưu ý khi lấy video trên mạng

Không nên push video tải từ mạng lên GitHub. Repo chỉ nên chứa code. Dataset/weights để Google Drive hoặc dùng video có giấy phép/được phép sử dụng cho học tập.
