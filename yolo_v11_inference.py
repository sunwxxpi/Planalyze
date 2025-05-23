import os
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

# GPU 사용 가능 여부 및 GPU 수 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Using device: {device}, Number of available GPUs: {num_gpus}")

# 다중 GPU 설정
gpu_devices = [i for i in range(num_gpus)] if num_gpus > 0 else None
device_str = ','.join(map(str, gpu_devices)) if gpu_devices else 'cpu'
print(f"Training on device(s): {device_str}")

# Background 클래스 제거
classes = [i for i in range(23) if i != 11]

model = YOLO(f"/home/psw/Planalyze/runs/segment/20250421_034638/exp/weights/best.pt")  # 훈련된 모델 로드

test_list = ["/home/psw/Planalyze/1_APT_FP_STR_024028684.png",
                "/home/psw/Planalyze/2_APT_FP_STR_029608817.png",
                "/home/psw/Planalyze/3_APT_FP_STR_030708405.png"]

# 테스트 이미지에 추론 실행
results = model(test_list,
                # imgsz=2528,
                save=True,
                show_boxes=False)

# 마스크 이미지만 따로 저장
mask_output_dir = "/home/psw/Planalyze/mask_outputs"
os.makedirs(mask_output_dir, exist_ok=True)

print(f"마스크 이미지를 {mask_output_dir}에 저장합니다.")

# 클래스별로 구분되는 색상 팔레트 정의 (BGR 형식)
color_palette = [
    (255, 0, 0),      # 파랑
    (0, 255, 0),      # 초록
    (0, 0, 255),      # 빨강
]

# 각 결과에서 마스크만 추출하여 저장
for i, result in enumerate(results):
    # 원본 이미지 경로에서 파일 이름만 추출
    filename = Path(test_list[i]).stem
    
    # 마스크 정보가 있는 경우에만 처리
    if result.masks is not None:
        # 이미지 크기 가져오기
        img_height, img_width = result.orig_shape
        
        # 빈 마스크 이미지 생성 (검은색 배경)
        mask_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # 각 클래스별 마스크를 처리
        for j, mask in enumerate(result.masks.data):
            # 클래스 정보 가져오기
            cls_id = int(result.boxes.cls[j].item())
            if cls_id in classes:  # Background 클래스인 11은 이미 제거되어 있음
                # 마스크를 numpy 배열로 변환하고 원본 이미지 크기로 조정
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (img_width, img_height))
                
                # 클래스별 색상 설정 (미리 정의된 팔레트에서 선택)
                color = color_palette[cls_id % len(color_palette)]
                
                # 마스크를 이미지에 적용
                mask_img[mask_np > 0.5] = color
        
        # 마스크 이미지 저장
        mask_path = f"{mask_output_dir}/{filename}_mask.png"
        cv2.imwrite(mask_path, mask_img)
        
        print(f"마스크 이미지 저장 완료: {mask_path}")

print("모든 마스크 이미지 저장이 완료되었습니다.")