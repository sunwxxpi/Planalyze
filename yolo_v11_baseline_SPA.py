import os
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

EPOCHS = 100
BATCH_SIZE = 2
IMAGE_SIZE = 2080
RESULT_DIR = f"./results/SPA/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TARGET_CLASSES = [0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22] # SPA Classes
GPU_DEVICE = 0

os.makedirs(RESULT_DIR, exist_ok=True)

# 사전 훈련된 YOLOv11 세그멘테이션 모델 사용
model = YOLO('yolo11s-seg.pt')

# 학습 설정
results = model.train(
                data="SPA.yaml",             # 데이터셋 설정 파일
                epochs=100,                  # 훈련 에폭 수
                batch=BATCH_SIZE,                     # 배치 크기
                imgsz=IMAGE_SIZE,            # 이미지 크기
                project=RESULT_DIR,          # 결과 저장 프로젝트 폴더
                val=True,                    # 검증 수행
                verbose=True,                # 상세 출력
                classes=TARGET_CLASSES,      # 클래스 설정
                device=GPU_DEVICE,
                # resume=True
)

model = YOLO(f"{RESULT_DIR}/train/weights/best.pt")  # 훈련된 모델 로드

# 학습 결과 평가
metrics = model.val(
                project=RESULT_DIR,
                imgsz=2080,
                verbose=True,
                classes=TARGET_CLASSES)  # 검증 데이터셋에 대한 평가

# 성능 결과를 저장할 리스트
results_lines = []

# SPA 클래스 이름 매핑 (SPA.yaml 기준)
class_names = {
    0: "공간_다목적공간", 1: "공간_엘리베이터홀", 2: "공간_계단실", 
    12: "공간_거실", 13: "공간_침실", 14: "공간_주방", 15: "공간_현관", 
    16: "공간_발코니", 17: "공간_화장실", 18: "공간_실외기실", 
    19: "공간_드레스룸", 21: "공간_기타", 22: "공간_엘리베이터"
}

# 전체 성능 출력
print("\n=== Overall Performance ===")
results_lines.append("=== Overall Performance ===")

print(f"Box Detection mAP50: {metrics.box.map50:.4f}")
print(f"Box Detection mAP50-95: {metrics.box.map:.4f}")
print(f"Segmentation mAP50: {metrics.seg.map50:.4f}")  
print(f"Segmentation mAP50-95: {metrics.seg.map:.4f}")

results_lines.append(f"Box Detection mAP50: {metrics.box.map50:.4f}")
results_lines.append(f"Box Detection mAP50-95: {metrics.box.map:.4f}")
results_lines.append(f"Segmentation mAP50: {metrics.seg.map50:.4f}")
results_lines.append(f"Segmentation mAP50-95: {metrics.seg.map:.4f}")

# 클래스별 성능 출력
print("\n=== Per-Class Segmentation Performance ===")
results_lines.append("\n=== Per-Class Segmentation Performance ===")

# 클래스별 mAP50, mAP 데이터 가져오기
seg_map50_per_class = getattr(metrics.seg, 'ap50', None)
seg_map_per_class = getattr(metrics.seg, 'ap', None)

if seg_map50_per_class is not None and len(seg_map50_per_class) >= len(TARGET_CLASSES):
    for i, class_id in enumerate(TARGET_CLASSES):
        class_name = class_names.get(class_id, f"class_{class_id}")
        map50_val = seg_map50_per_class[i] if seg_map50_per_class[i] is not None else 0.0
        map_val = seg_map_per_class[i] if seg_map_per_class is not None and i < len(seg_map_per_class) and seg_map_per_class[i] is not None else 0.0
        
        print(f"  {class_name} (ID: {class_id})")
        print(f"    - mAP50: {map50_val:.4f}")
        print(f"    - mAP50-95: {map_val:.4f}")
        
        results_lines.append(f"  {class_name} (ID: {class_id})")
        results_lines.append(f"    - mAP50: {map50_val:.4f}")
        results_lines.append(f"    - mAP50-95: {map_val:.4f}")
else:
    print("No per-class segmentation metrics available")
    results_lines.append("No per-class segmentation metrics available")

# 추가 통계 정보
print("\n=== Additional Statistics ===")
results_lines.append("\n=== Additional Statistics ===")

if hasattr(metrics.seg, 'mp'):
    print(f"Segmentation Precision: {metrics.seg.mp:.4f}")
    results_lines.append(f"Segmentation Precision: {metrics.seg.mp:.4f}")
    
if hasattr(metrics.seg, 'mr'):
    print(f"Segmentation Recall: {metrics.seg.mr:.4f}")
    results_lines.append(f"Segmentation Recall: {metrics.seg.mr:.4f}")

# results.txt 파일로 저장
results_file_path = os.path.join(RESULT_DIR, "results.txt")
with open(results_file_path, 'w', encoding='utf-8') as f:
    f.write(f"YOLO v11 SPA Segmentation Performance Results\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model path: {RESULT_DIR}/train/weights/best.pt\n")
    f.write(f"Number of classes: {len(TARGET_CLASSES)}\n\n")
    
    for line in results_lines:
        f.write(line + '\n')

print(f"Training completed. Results saved to {RESULT_DIR}")