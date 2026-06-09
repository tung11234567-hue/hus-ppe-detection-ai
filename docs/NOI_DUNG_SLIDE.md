# Nội dung slide thuyết trình

## Slide 1: Tiêu đề

Hệ thống AI phát hiện người không đeo bảo hộ lao động tại công trường

Thành viên nhóm, lớp, môn học, giảng viên.

## Slide 2: Lý do chọn đề tài

- Công trường có nhiều rủi ro tai nạn lao động.
- Mũ bảo hộ và áo phản quang là trang bị quan trọng.
- Giám sát thủ công tốn nhân lực và dễ bỏ sót.
- AI có thể hỗ trợ phát hiện tự động qua camera.

## Slide 3: Mục tiêu

- Nhận diện người, mũ bảo hộ, áo phản quang.
- Phát hiện người thiếu mũ hoặc thiếu áo phản quang.
- Chạy được trên ảnh, video và webcam.
- Có giao diện demo dễ sử dụng.

## Slide 4: Bài toán AI

Input: ảnh/video công trường.

Output:

- Bounding box người.
- Bounding box mũ/áo.
- Nhãn cảnh báo: SAFE, NO_HELMET, NO_VEST.

## Slide 5: Dataset

- Định dạng YOLO.
- Các class: person, helmet, safety_vest.
- Chia train/valid/test.
- Có augmentation nếu dùng Roboflow hoặc công cụ tương tự.

## Slide 6: Mô hình YOLO

- YOLO là mô hình Object Detection một giai đoạn.
- Ưu điểm: tốc độ nhanh, phù hợp real-time.
- Fine-tune từ pretrained model để giảm thời gian train.

## Slide 7: Pipeline hệ thống

1. Camera/ảnh/video.
2. YOLO detection.
3. Ghép PPE với từng người.
4. Kiểm tra vùng đầu/vùng thân.
5. Hiển thị cảnh báo.

## Slide 8: Luật phát hiện vi phạm

- Nếu vùng đầu của người không có mũ: NO_HELMET.
- Nếu vùng thân của người không có áo phản quang: NO_VEST.
- Nếu có đủ cả hai: SAFE.

## Slide 9: Demo giao diện

Chụp màn hình app Streamlit hoặc video demo.

## Slide 10: Kết quả

Điền sau khi train:

- Precision
- Recall
- mAP50
- mAP50-95
- FPS

## Slide 11: Hạn chế

- Che khuất, ánh sáng yếu.
- Người ở xa camera.
- Dataset chưa đủ đa dạng.
- Dễ nhầm áo phản quang với áo màu sáng.

## Slide 12: Hướng phát triển

- Thêm class kính/găng tay/khẩu trang.
- Tích hợp cảnh báo real-time.
- Chạy trên thiết bị biên như Jetson/Raspberry Pi.
- Thu thập thêm dữ liệu thực tế.
