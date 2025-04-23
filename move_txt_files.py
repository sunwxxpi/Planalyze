import os
import shutil
import sys

def move_txt_files(train_dir):
    """
    train 디렉토리 내의 모든 하위 디렉토리에서 txt 파일을 찾아
    train 디렉토리로 직접 이동시키는 함수
    """
    # train 디렉토리가 존재하는지 확인
    if not os.path.isdir(train_dir):
        print(f"오류: {train_dir} 디렉토리가 존재하지 않습니다.")
        return

    # 모든 하위 디렉토리 검색
    subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
               if os.path.isdir(os.path.join(train_dir, d))]
    
    moved_count = 0
    for subdir in subdirs:
        # 각 하위 디렉토리 내의 txt 파일 검색
        for root, _, files in os.walk(subdir):
            for file in files:
                if file.endswith('.txt'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(train_dir, file)
                    
                    # 파일명 충돌 처리
                    if os.path.exists(dst_path):
                        base_name, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(os.path.join(train_dir, f"{base_name}_{counter}{ext}")):
                            counter += 1
                        dst_path = os.path.join(train_dir, f"{base_name}_{counter}{ext}")
                    
                    # 파일 이동
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                    print(f"이동됨: {src_path} -> {dst_path}")
    
    print(f"총 {moved_count}개의 txt 파일이 이동되었습니다.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_directory = sys.argv[1]
    else:
        train_directory = os.path.join(os.getcwd(), 'yolo_drawing_data/labels/train')
    
    print(f"'{train_directory}' 디렉토리 내의 모든 txt 파일을 이동합니다...")
    move_txt_files(train_directory)