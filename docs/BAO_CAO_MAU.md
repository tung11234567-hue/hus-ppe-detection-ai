# Báo cáo mẫu: Hệ thống AI phát hiện người không đeo bảo hộ lao động tại công trường

## 1. Giới thiệu đề tài

Trong môi trường công trường, việc không đội mũ bảo hộ hoặc không mặc áo phản quang có thể gây nguy hiểm cho người lao động. Đề tài xây dựng một hệ thống trí tuệ nhân tạo có khả năng tự động phát hiện người lao động thiếu trang bị bảo hộ thông qua ảnh, video hoặc camera.

## 2. Mục tiêu

- Phát hiện người trong ảnh/video công trường.
- Phát hiện mũ bảo hộ và áo phản quang.
- Xác định người nào đang thiếu mũ hoặc thiếu áo phản quang.
- Hiển thị kết quả trực quan bằng bounding box và nhãn cảnh báo.

## 3. Cơ sở lý thuyết

### 3.1. Object Detection

Object Detection là bài toán vừa phân loại đối tượng vừa xác định vị trí của đối tượng bằng bounding box. Trong đề tài này, mỗi đối tượng được biểu diễn bởi:

- Tọa độ hộp: `(x1, y1, x2, y2)`
- Nhãn lớp: `person`, `helmet`, `safety_vest`
- Độ tin cậy: `confidence score`

### 3.2. YOLO

YOLO là nhóm mô hình phát hiện vật thể một giai đoạn, có tốc độ nhanh nên phù hợp cho bài toán real-time. Ảnh đầu vào được đưa qua mạng CNN, sau đó mô hình dự đoán bounding box, class và confidence.

### 3.3. Transfer Learning

Thay vì train từ đầu, nhóm dùng mô hình YOLO đã được pretrain rồi fine-tune trên dataset bảo hộ lao động. Cách này giúp giảm thời gian huấn luyện và cần ít dữ liệu hơn.

## 4. Dataset

Dataset gồm ảnh công trường có gán nhãn các đối tượng:

- `person`: người/công nhân
- `helmet`: mũ bảo hộ
- `safety_vest`: áo phản quang

Dữ liệu được chia thành:

- Train set: dùng để huấn luyện
- Validation set: dùng để chọn mô hình tốt nhất
- Test set: dùng để kiểm tra cuối cùng

## 5. Phương pháp đề xuất

Hệ thống gồm hai bước chính:

### Bước 1: Phát hiện đối tượng bằng YOLO

Mô hình YOLO nhận ảnh đầu vào và trả về danh sách bounding box của người, mũ và áo phản quang.

### Bước 2: Kiểm tra vi phạm bảo hộ

Với mỗi bounding box người:

- Chia vùng người thành vùng đầu và vùng thân.
- Nếu vùng đầu không có bounding box mũ thì kết luận `NO_HELMET`.
- Nếu vùng thân không có bounding box áo phản quang thì kết luận `NO_VEST`.

## 6. Cài đặt thực nghiệm

Môi trường:

- Python 3.10+
- Ultralytics YOLO
- OpenCV
- Streamlit

Lệnh train:

```bash
python train.py --data data/ppe.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --batch 16
```

Lệnh chạy demo:

```bash
streamlit run app_streamlit.py
```

## 7. Kết quả đánh giá

Bảng kết quả cần điền sau khi train:

| Chỉ số | Giá trị |
|---|---:|
| Precision | ... |
| Recall | ... |
| mAP50 | ... |
| mAP50-95 | ... |
| FPS webcam | ... |

Nhận xét:

- Model nhận diện tốt khi ảnh rõ, người đứng gần camera.
- Model dễ nhầm khi người bị che khuất, mũ/áo quá nhỏ hoặc ánh sáng yếu.

## 8. Kết luận

Đề tài đã xây dựng được hệ thống phát hiện vi phạm bảo hộ lao động tại công trường. Hệ thống có thể nhận ảnh, video hoặc webcam, sau đó đưa ra cảnh báo nếu công nhân không đội mũ hoặc không mặc áo phản quang.

## 9. Hướng phát triển

- Bổ sung thêm găng tay, kính bảo hộ, khẩu trang.
- Tăng dữ liệu ở công trường thực tế tại Việt Nam.
- Tối ưu model để chạy trên camera biên hoặc Raspberry Pi/Jetson Nano.
- Tích hợp gửi cảnh báo qua Telegram/email khi phát hiện vi phạm.
