@echo off
REM 1) Put videos into raw_videos\ first
REM 2) Activate venv before running this file
python scripts\extract_frames.py --input raw_videos --output datasets\ppe_raw\all\images --every-sec 1
python scripts\split_yolo_dataset.py --src datasets\ppe_raw\all --dst datasets\ppe --require-labels --clean
python scripts\check_yolo_dataset.py --data data\ppe.yaml
python train.py --data data\ppe.yaml --model yolo11n.pt --epochs 80 --imgsz 640 --batch 16
