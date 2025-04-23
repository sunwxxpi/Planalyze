import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO

# GPU 사용 가능 여부 및 GPU 수 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Using device: {device}, Number of available GPUs: {num_gpus}")

# 다중 GPU 설정
gpu_devices = [i for i in range(num_gpus)] if num_gpus > 0 else None
device_str = ','.join(map(str, gpu_devices)) if gpu_devices else 'cpu'
print(f"Training on device(s): {device_str}")

# 저장 경로 설정
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/home/psw/Planalyze/runs/segment/{current_time}"
os.makedirs(save_dir, exist_ok=True)

# 모델 로드
# 방법 1: 사전 훈련된 YOLOv11 세그멘테이션 모델 사용
model = YOLO('yolo11n-seg.pt')
# model = YOLO('runs/segment/20250421_034638/exp/weights/last.pt')

# Background 클래스 제거
classes = [i for i in range(23) if i != 11]

# 학습 설정
results = model.train(
    data="STR.yaml",           # 데이터셋 설정 파일
    epochs=100,                # 훈련 에폭 수
    batch=4,                   # 배치 크기
    imgsz=1536,                # 이미지 크기
    project=save_dir,          # 결과 저장 프로젝트 폴더
    name="exp",                # 결과 저장 이름
    val=True,                  # 검증 수행
    rect=False,                # 직사각형 학습 설정
    verbose=True,              # 상세 출력
    device=device_str,         # 다중 GPU 설정
    classes=classes,
    # resume=True
)

# model = YOLO(f"{save_dir}/exp/weights/best.pt")  # 훈련된 모델 로드
model = YOLO(f"{save_dir}/exp/weights/best.pt")  # 훈련된 모델 로드

# 학습 결과 평가
metrics = model.val(classes=classes)  # 검증 데이터셋에 대한 평가
print("Metrics:", metrics.seg.maps)

# 테스트 이미지에 추론 실행 (선택적)
# results = model("path/to/test/images")

print(f"Training completed. Results saved to {save_dir}")