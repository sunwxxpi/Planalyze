import os
from datetime import datetime
from ultralytics import YOLO

# 저장 경로 설정
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"./runs/segment/STR/{current_time}"
os.makedirs(save_dir, exist_ok=True)

# 사전 훈련된 YOLOv11 세그멘테이션 모델 사용
model = YOLO('yolo11s-seg.pt')

# STR Classes
classes = [8, 9, 10]

# 학습 설정
results = model.train(
    data="STR.yaml",           # 데이터셋 설정 파일
    epochs=100,                # 훈련 에폭 수
    # batch=4,                   # 배치 크기
    # imgsz=2112,                # 이미지 크기
    batch=2,                   # 배치 크기
    imgsz=2400,                # 이미지 크기
    project=save_dir,          # 결과 저장 프로젝트 폴더
    name="exp/STR",            # 결과 저장 이름
    val=True,                  # 검증 수행
    rect=False,                # 직사각형 학습 설정
    verbose=True,              # 상세 출력
    classes=classes,           # 클래스 설정
    # resume=True
)

model = YOLO(f"{save_dir}/exp/STR/weights/best.pt")  # 훈련된 모델 로드

# 학습 결과 평가
metrics = model.val(classes=classes)  # 검증 데이터셋에 대한 평가
print("Metrics:", metrics.seg.maps)

""" test_list = ["./1_APT_FP_STR_024028684.png",
                "./2_APT_FP_STR_029608817.png",
                "./3_APT_FP_STR_030708405.png"]

# 테스트 이미지에 추론 실행
results = model(test_list,
                save=True,
                show_boxes=False) """

print(f"Training completed. Results saved to {save_dir}")