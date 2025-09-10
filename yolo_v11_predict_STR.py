import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# 프로젝트 설정
RESULT_DIR = "/home/work/.Planalyze/Planalyze/results/STR/20250907_020102"
TARGET_CLASSES = [8, 9, 10]
IMAGE_SIZE = 2080

# 클래스별 색상 (BGR 형식)
COLORS = [
    (255, 0, 0),    # 파란색 (Blue)
    (0, 255, 0),    # 초록색 (Green)
    (0, 0, 255),    # 빨간색 (Red)
]

# 테스트 이미지 목록
test_images = [
    "./APT_FP_STR_024028684.png",
    "./APT_FP_STR_029608817.png", 
    "./APT_FP_STR_030708405.png"
]

# 훈련된 YOLO 모델 로드
model = YOLO(f"{RESULT_DIR}/train/weights/best.pt")

# 이미지 추론 실행
print(f"{len(test_images)}개 이미지에 대해 추론을 실행합니다...")
results = model.predict(
    test_images,
    imgsz=IMAGE_SIZE,
    device=1
)

# 결과 저장 디렉토리 생성
mask_dir = f"{RESULT_DIR}/predict/mask_outputs"
predict_dir = f"{RESULT_DIR}/predict"
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(predict_dir, exist_ok=True)

# 각 이미지의 추론 결과 처리
for i, result in enumerate(results):
    # 파일명 추출
    filename = Path(test_images[i]).stem
    
    # 원본 이미지 로드
    original_image = cv2.imread(test_images[i])
    
    # 마스크가 있는 경우에만 처리
    if result.masks is not None:
        # 원본 이미지 크기
        height, width = result.orig_shape
        
        # 원본 이미지를 올바른 크기로 리사이즈
        if original_image.shape[:2] != (height, width):
            original_image = cv2.resize(original_image, (width, height))
        
        # 색상 마스크를 위한 빈 이미지 생성 (검은 배경)
        mask_only_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 색상 오버레이를 위한 빈 이미지 생성
        color_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 각 탐지된 객체의 마스크 처리
        for j, mask in enumerate(result.masks.data):
            class_id = int(result.boxes.cls[j].item())
            
            # 지정된 클래스에 해당하는 경우만 처리
            if class_id in TARGET_CLASSES:
                # 마스크를 원본 크기로 변환
                mask_array = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_array, (width, height))
                
                # TARGET_CLASSES에서의 인덱스를 사용하여 색상 선택
                color_index = TARGET_CLASSES.index(class_id)
                color = COLORS[color_index % len(COLORS)]
                
                # 마스크 픽셀 위치
                mask_pixels = mask_resized > 0.5
                
                # 색상 마스크에 해당 클래스 색상 적용
                mask_only_image[mask_pixels] = color
                
                # 색상 오버레이에 해당 클래스 색상 적용
                color_overlay[mask_pixels] = color
        
        # 원본과 색상 오버레이를 블렌딩 (투명도 0.4)
        alpha = 0.4
        blended_image = cv2.addWeighted(original_image, 1-alpha, color_overlay, alpha, 0)
        
        # 1. predict 디렉토리에 오버레이 이미지 저장
        predict_output_path = f"{predict_dir}/{filename}.png"
        cv2.imwrite(predict_output_path, blended_image)
        
        # 2. mask_outputs 디렉토리에 색상 마스크 저장
        mask_output_path = f"{mask_dir}/{filename}.png"
        cv2.imwrite(mask_output_path, mask_only_image)

print("\n모든 마스크 이미지 생성 및 저장이 완료되었습니다.")