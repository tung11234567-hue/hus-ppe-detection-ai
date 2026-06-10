# HUS PPE Detection AI

## Giới thiệu

Đây là project cuối kỳ môn Nhập môn Trí tuệ nhân tạo, xây dựng hệ thống AI phát hiện vi phạm bảo hộ lao động tại công trường.

Hệ thống sử dụng mô hình YOLO26x để nhận diện người lao động và các thiết bị bảo hộ như mũ bảo hộ, áo phản quang. Sau khi phát hiện đối tượng, chương trình tiếp tục phân tích quan hệ giữa người và đồ bảo hộ để suy luận trạng thái an toàn hoặc vi phạm.

Mục tiêu của project là hỗ trợ giám sát an toàn lao động trong môi trường công trường thông qua ảnh, video hoặc webcam.

## Thông tin môn học

- Môn học: Nhập môn Trí tuệ nhân tạo
- Trường: Trường Đại học Khoa học Tự nhiên, ĐHQGHN (VNU-HUS)
- Giảng viên: CN. Vi Anh Quân, GS. Nguyễn Thế Toàn
- Đề tài: Phát hiện vi phạm bảo hộ lao động tại công trường bằng YOLO26x

## Thành viên nhóm

| STT | Họ và tên | Công việc | Đóng góp |
|---|---|---|---|
| 1 | Phạm Đức Anh | Viết báo cáo, làm slide | 25% |
| 2 | Đặng Tùng Anh | Hỗ trợ code, train AI | 25% |
| 3 | Nguyễn Hà Gia Bảo | Thiết kế code, web demo | 25% |
| 4 | Vũ Việt Anh | Viết báo cáo, làm slide | 25% |

## Bài toán

Trong môi trường công trường, người lao động cần tuân thủ các quy định bảo hộ cơ bản như:

* Phải đội mũ bảo hộ
* Phải mặc áo phản quang
* Cần được phát hiện vi phạm theo thời gian thực hoặc gần thời gian thực

Project này giải quyết bài toán bằng cách phát hiện các đối tượng liên quan đến PPE và suy luận trạng thái của từng người xuất hiện trong khung hình.

## Chức năng chính

* Phát hiện person, helmet, safety_vest, no_helmet, no_vest
* Phân tích trạng thái từng người: SAFE, NO_HELMET, NO_VEST
* Hỗ trợ nhận diện trên ảnh
* Hỗ trợ nhận diện trên video
* Hỗ trợ demo qua giao diện web Streamlit
* Hỗ trợ webcam
* Có tracking ID cho person trong video
* Hiển thị bounding box, nhãn đối tượng và trạng thái PPE
* Có thể lưu kết quả đầu ra phục vụ demo và báo cáo

## Công nghệ sử dụng

* Python
* Ultralytics YOLO
* YOLO26x
* OpenCV
* Streamlit
* NumPy
* PyYAML

## Mô hình AI

Project sử dụng YOLO26x cho bài toán object detection.

YOLO là mô hình phát hiện đối tượng một giai đoạn. Thay vì tách riêng bước đề xuất vùng và phân loại, YOLO dự đoán trực tiếp bounding box, class và confidence score trên ảnh đầu vào.

Các thông tin mô hình sử dụng:

* Model: YOLO26x
* Framework: Ultralytics
* Dataset format: YOLOv8 / Ultralytics YOLO
* Input size khi train: 768
* Bài toán: Object Detection
* Số class sau xử lý: 5

## Danh sách class

Dataset sau khi xử lý được chuẩn hóa về 5 class:

* 0: person
* 1: helmet
* 2: safety_vest
* 3: no_helmet
* 4: no_vest

## Thuật toán xử lý

Pipeline xử lý chính của hệ thống gồm các bước:

1. Đọc ảnh hoặc từng frame từ video/webcam
2. Đưa ảnh vào mô hình YOLO26x để phát hiện đối tượng
3. Lọc kết quả dựa trên confidence score
4. Sử dụng IoU và Non-Maximum Suppression để giảm box trùng
5. Tách các đối tượng phát hiện được theo class
6. Ghép helmet và safety_vest với từng person dựa trên vị trí bounding box
7. Suy luận trạng thái PPE của từng người
8. Vẽ bounding box, nhãn class, trạng thái an toàn và tracking ID
9. Hiển thị hoặc lưu kết quả đầu ra

## Confidence Score

Confidence score thể hiện độ tin cậy của mô hình với một đối tượng được phát hiện.

Ví dụ:

* Confidence cao: mô hình khá chắc chắn đối tượng đó là person, helmet hoặc safety_vest
* Confidence thấp: mô hình chưa chắc chắn, dễ gây nhận diện sai

Trong project, có thể chỉnh ngưỡng confidence bằng tham số:

```
--conf 0.05
```

## IoU

IoU là Intersection over Union, dùng để đo mức độ chồng lấn giữa hai bounding box.

IoU được sử dụng trong quá trình lọc box trùng. Nếu nhiều box cùng phát hiện một đối tượng, hệ thống sẽ giữ box có confidence cao hơn và loại bỏ box dư thừa.

## Non-Maximum Suppression

Non-Maximum Suppression là kỹ thuật loại bỏ các bounding box trùng nhau sau khi mô hình dự đoán.

Trong bài toán PPE detection, NMS giúp tránh trường hợp một người hoặc một chiếc mũ bị vẽ nhiều khung phát hiện cùng lúc.

## Suy luận trạng thái PPE

Sau khi YOLO phát hiện các đối tượng, hệ thống không chỉ hiển thị class riêng lẻ mà còn phân tích trạng thái bảo hộ của từng người.

Logic suy luận:

* Nếu person có helmet và safety_vest phù hợp: SAFE
* Nếu person thiếu helmet: NO_HELMET
* Nếu person thiếu safety_vest: NO_VEST
* Nếu thiếu cả hai: NO_HELMET và NO_VEST

Việc ghép đồ bảo hộ với person dựa trên vị trí tương đối của bounding box trong ảnh.

## Tracking ID

Với video, hệ thống có thể gán ID cho từng person để dễ theo dõi người đó qua nhiều frame.

Ví dụ:

* Person ID 1
* Person ID 2
* Person ID 3

Tracking ID giúp phần demo trực quan hơn, tránh chỉ đếm người theo từng frame rời rạc.

## Dataset

Dataset ban đầu có nhiều class khác nhau liên quan đến người và bảo hộ lao động. Trước khi train, dataset được xử lý và chuẩn hóa về 5 class chính.

Dataset gốc gồm 16 class, sau đó được convert về 5 class:

* Human -> person
* Helmet, hat -> helmet
* Safety Vest, vest -> safety_vest
* no hat -> no_helmet
* no vest -> no_vest

Một dataset bổ sung về công nhân xây dựng cũng được chuyển đổi về cùng format để tăng dữ liệu train cho các class person, helmet và safety_vest.

## Kết quả train

Kết quả tốt nhất hiện tại:

* Best mAP50: 0.73361
* Model tốt nhất được lưu ở file best.pt
* Model cuối quá trình train được lưu ở file last.pt

Lưu ý: file best.pt không được upload trực tiếp lên GitHub vì dung lượng lớn.

## Cấu trúc project

```
hus-ppe-detection-ai
├── app_streamlit.py
├── detect.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
├── configs
├── docs
├── raw_videos
├── scripts
├── src
│   └── ppe_detector
├── test_media
├── tests
└── weights
```

## Cài đặt project

Clone project về máy:

```
git clone https://github.com/tung11234567-hue/hus-ppe-detection-ai.git
```

Di chuyển vào thư mục project:

```
cd hus-ppe-detection-ai
```

Tạo môi trường ảo:

```
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Cài thư viện:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model weights

Không upload best.pt lên GitHub vì file model nặng.

Sau khi tải model, đặt file vào thư mục:

```
weights\best.pt
```

Kiểm tra model:

```
dir weights
```

Trong thư mục weights cần có:

```
best.pt
```

## Chạy web demo

Chạy giao diện Streamlit:

```
streamlit run app_streamlit.py
```

Sau đó mở link local mà Streamlit hiển thị trên terminal.

## Test ảnh

Chạy nhận diện trên ảnh:

```
python detect.py --weights weights\best.pt --source "test_media\test1.jpg" --view --conf 0.05
```

## Test video

Chạy nhận diện trên video:

```
python detect.py --weights weights\best.pt --source "raw_videos\test2.mp4" --view --conf 0.05
```

## Test webcam

Nếu chương trình hỗ trợ webcam, có thể chạy source bằng camera:

```
python detect.py --weights weights\best.pt --source 0 --view --conf 0.05
```

## Huấn luyện mô hình

Quá trình train được thực hiện bằng Ultralytics YOLO.

Ví dụ lệnh train:

```
yolo detect train model=yolo26x.pt data=ppe5.yaml epochs=20 imgsz=768 batch=16 device=0
```

Có thể train tiếp từ model đã lưu:

```
yolo detect train model=best.pt data=ppe5.yaml epochs=20 imgsz=768 batch=16 device=0
```

## File cấu hình dataset

Ví dụ file ppe5.yaml:

```
path: /kaggle/working/ppe5_correct
train: train/images
val: valid/images
test: test/images

names:
  0: person
  1: helmet
  2: safety_vest
  3: no_helmet
  4: no_vest
```

## Ý nghĩa project

Project mô phỏng một hệ thống giám sát an toàn lao động bằng AI. Hệ thống có thể hỗ trợ phát hiện người không đội mũ bảo hộ hoặc không mặc áo phản quang trong khu vực công trường.

Ứng dụng thực tế có thể mở rộng cho:

* Camera giám sát công trường
* Hệ thống cảnh báo an toàn lao động
* Kiểm tra tuân thủ PPE trong nhà máy
* Phân tích video giám sát sau sự kiện

## Hạn chế

* Kết quả phụ thuộc vào chất lượng dataset
* Có thể nhận diện sai khi người bị che khuất
* Ánh sáng yếu hoặc góc quay xấu có thể làm giảm độ chính xác
* Một số trường hợp helmet hoặc vest bị che khuất có thể gây suy luận sai trạng thái
* Chưa triển khai cảnh báo thời gian thực hoàn chỉnh

## Hướng phát triển

* Tăng thêm dữ liệu thực tế từ công trường
* Bổ sung các class khác như gloves, boots, glasses
* Cải thiện tracking ID cho video đông người
* Tối ưu tốc độ xử lý real-time
* Thêm cảnh báo âm thanh hoặc gửi thông báo khi phát hiện vi phạm
* Triển khai trên camera IP hoặc hệ thống giám sát thực tế

## Tác giả

Nhóm thực hiện project cuối kỳ môn Nhập môn Trí tuệ nhân tạo.

Thành viên:

* Phạm Đức Anh
* Đặng Tùng Anh
* Nguyễn Hà Gia Bảo
* Vũ Việt Anh
