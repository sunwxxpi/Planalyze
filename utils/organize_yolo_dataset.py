import shutil
from pathlib import Path

def reorganize_yolo_dataset(source_dir, target_dir):
    """
    데이터셋을 YOLO 학습용 구조로 재구성
    
    Args:
        source_dir: 현재 데이터셋 경로 (datasets 폴더)
        target_dir: 새로운 데이터셋 경로
    """
    
    # 타겟 디렉토리 생성
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # STR과 SPA(TS_SPA) 구조 생성
    for class_type in ['STR', 'SPA']:
        for data_type in ['images', 'labels']:
            for split in ['train', 'val']:
                (target_path / class_type / data_type / split).mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Training 데이터 처리 - 모두 train으로
    training_path = source_path / 'Training'
    if training_path.exists():
        print(f"Training 폴더 발견: {training_path}")
        process_data_split(training_path, target_path, 'train')
    else:
        print(f"Training 폴더가 없습니다: {training_path}")
    
    # Validation 데이터 처리 - 모두 val로
    validation_path = source_path / 'Validation'
    if validation_path.exists():
        print(f"Validation 폴더 발견: {validation_path}")
        process_data_split(validation_path, target_path, 'val')
    else:
        print(f"Validation 폴더가 없습니다: {validation_path}")

def process_data_split(data_path, target_path, split_type):
    """
    데이터 분할 처리
    """
    
    print(f"  처리 중: {data_path} → {split_type}")
    
    # 이미지 처리
    images_path = data_path / 'images'
    if images_path.exists():
        print(f"    이미지 폴더 발견: {images_path}")
        process_images(images_path, target_path, split_type)
    else:
        print(f"    이미지 폴더가 없습니다: {images_path}")
    
    # 라벨 처리
    labels_path = data_path / 'labels'
    if labels_path.exists():
        print(f"    라벨 폴더 발견: {labels_path}")
        process_labels(labels_path, target_path, split_type)
    else:
        print(f"    라벨 폴더가 없습니다: {labels_path}")

def process_images(images_path, target_path, split_type):
    """
    이미지 파일 처리 및 복사
    """
    
    print(f"      이미지 폴더 내용: {list(images_path.iterdir())}")
    
    # SPA 클래스 통합 (TS_SPA_1, TS_SPA_2 -> SPA)
    spa_folders = [f for f in images_path.iterdir() if f.is_dir() and 'SPA' in f.name]
    str_folders = [f for f in images_path.iterdir() if f.is_dir() and 'STR' in f.name]
    
    print(f"      SPA 폴더들: {spa_folders}")
    print(f"      STR 폴더들: {str_folders}")
    
    # SPA 데이터 처리
    for spa_folder in spa_folders:
        print(f"        SPA 폴더 처리: {spa_folder}")
        image_files = list(spa_folder.glob('*'))
        print(f"          이미지 파일 수: {len(image_files)}")
        for img_file in image_files:
            if img_file.is_file():
                target_file = target_path / 'SPA' / 'images' / split_type / img_file.name
                print(f"          복사: {img_file} → {target_file}")
                shutil.copy2(img_file, target_file)
    
    # STR 데이터 처리
    for str_folder in str_folders:
        print(f"        STR 폴더 처리: {str_folder}")
        image_files = list(str_folder.glob('*'))
        print(f"          이미지 파일 수: {len(image_files)}")
        for img_file in image_files:
            if img_file.is_file():
                target_file = target_path / 'STR' / 'images' / split_type / img_file.name
                print(f"          복사: {img_file} → {target_file}")
                shutil.copy2(img_file, target_file)

def process_labels(labels_path, target_path, split_type):
    """
    라벨 파일 처리 및 복사 (중간 디렉토리 포함)
    """
    
    print(f"      라벨 폴더 내용: {list(labels_path.iterdir())}")
    
    # SPA 라벨 통합
    spa_labels = [f for f in labels_path.iterdir() if f.is_dir() and 'SPA' in f.name]
    str_labels = [f for f in labels_path.iterdir() if f.is_dir() and 'STR' in f.name]
    
    print(f"      SPA 라벨 폴더들: {spa_labels}")
    print(f"      STR 라벨 폴더들: {str_labels}")
    
    # SPA 라벨 처리
    for spa_folder in spa_labels:
        print(f"        SPA 라벨 폴더 처리: {spa_folder}")
        # 중간 디렉토리들 탐색 (APT_CS_STR_000856837 같은 폴더들)
        for sub_folder in spa_folder.iterdir():
            if sub_folder.is_dir():
                print(f"          하위 폴더 발견: {sub_folder}")
                label_files = list(sub_folder.glob('*.txt'))
                print(f"            txt 파일 수: {len(label_files)}")
                for label_file in label_files:
                    if label_file.is_file():
                        target_file = target_path / 'SPA' / 'labels' / split_type / label_file.name
                        print(f"            복사: {label_file} → {target_file}")
                        shutil.copy2(label_file, target_file)
        
        # 직접 파일도 확인 (중간 디렉토리 없는 경우)
        direct_files = list(spa_folder.glob('*.txt'))
        if direct_files:
            print(f"          직접 txt 파일 수: {len(direct_files)}")
            for label_file in direct_files:
                target_file = target_path / 'SPA' / 'labels' / split_type / label_file.name
                print(f"          복사: {label_file} → {target_file}")
                shutil.copy2(label_file, target_file)
    
    # STR 라벨 처리
    for str_folder in str_labels:
        print(f"        STR 라벨 폴더 처리: {str_folder}")
        # 중간 디렉토리들 탐색 (APT_CS_STR_000856837 같은 폴더들)
        for sub_folder in str_folder.iterdir():
            if sub_folder.is_dir():
                print(f"          하위 폴더 발견: {sub_folder}")
                label_files = list(sub_folder.glob('*.txt'))
                print(f"            txt 파일 수: {len(label_files)}")
                for label_file in label_files:
                    if label_file.is_file():
                        target_file = target_path / 'STR' / 'labels' / split_type / label_file.name
                        print(f"            복사: {label_file} → {target_file}")
                        shutil.copy2(label_file, target_file)
        
        # 직접 파일도 확인 (중간 디렉토리 없는 경우)
        direct_files = list(str_folder.glob('*.txt'))
        if direct_files:
            print(f"          직접 txt 파일 수: {len(direct_files)}")
            for label_file in direct_files:
                target_file = target_path / 'STR' / 'labels' / split_type / label_file.name
                print(f"          복사: {label_file} → {target_file}")
                shutil.copy2(label_file, target_file)

def main():
    # 사용자 설정
    source_directory = "./dataset"  # 현재 데이터셋 경로
    target_directory = "./yolo_dataset"  # 새로운 YOLO 데이터셋 경로
    
    print("데이터셋 구조 변환을 시작합니다...")
    print(f"소스 디렉토리: {source_directory}")
    print(f"타겟 디렉토리: {target_directory}")
    print("Training 폴더 → train, Validation 폴더 → val")
    
    reorganize_yolo_dataset(source_directory, target_directory)
    
    print("데이터셋 구조 변환이 완료되었습니다!")
    print("\n생성된 구조:")
    print("yolo_dataset/")
    print("├── STR/")
    print("│   ├── images/")
    print("│   │   ├── train/ (Training 폴더의 STR 데이터)")
    print("│   │   └── val/ (Validation 폴더의 STR 데이터)")
    print("│   └── labels/")
    print("│       ├── train/ (Training 폴더의 STR 라벨)")
    print("│       └── val/ (Validation 폴더의 STR 라벨)")
    print("└── SPA/")
    print("    ├── images/")
    print("    │   ├── train/ (Training 폴더의 SPA 데이터)")
    print("    │   └── val/ (Validation 폴더의 SPA 데이터)")
    print("    └── labels/")
    print("        ├── train/ (Training 폴더의 SPA 라벨)")
    print("        └── val/ (Validation 폴더의 SPA 라벨)")

if __name__ == "__main__":
    main()