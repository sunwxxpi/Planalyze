import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from ultralytics import YOLO

model_path_str = "/home/psw/Planalyze/runs/segment/20250421_034638/exp/weights/best.pt"  # str 모델
model_path_spa = "./yolov8n-best-spa.pt"  # spa 모델

# 색상 팔레트 정의 (클래스 수만큼 생성)
def generate_palette(num_classes):
    np.random.seed(42) # 일관된 색상 팔레트 생성을 위해 시드 고정
    return [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(num_classes)]

# 마스크 이미지 생성 함수 (RGBA, 투명 배경)
def generate_mask(result, model_specific_names_dict, class_name_to_global_color_map): # excluded_class_id 제거
    if result.masks is None or len(result.masks.data) == 0:
        return None

    h, w = result.orig_shape
    mask_img_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for i, mask_tensor in enumerate(result.masks.data):
        cls_id = int(result.boxes.cls[i].item())

        class_name = model_specific_names_dict.get(cls_id)
        if class_name is None:
            continue

        # 통합 색상 맵에서 클래스 이름으로 색상 조회
        # 맵에 이름이 없으면 (필터링된 클래스) color는 None이 됨
        color = class_name_to_global_color_map.get(class_name)

        if color is None: # 필터링되어 색상이 할당되지 않은 클래스는 건너뜀
            continue

        mask_np = mask_tensor.cpu().numpy()
        mask_np_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        segment_pixels = mask_np_resized > 0.5
        mask_img_rgba[segment_pixels, 0] = color[0]
        mask_img_rgba[segment_pixels, 1] = color[1]
        mask_img_rgba[segment_pixels, 2] = color[2]
        mask_img_rgba[segment_pixels, 3] = 255

    if np.any(mask_img_rgba[:, :, 3] > 0):
        return Image.fromarray(mask_img_rgba, 'RGBA')
    else:
        return None

# RGBA 마스크에 사용자 알파 값을 적용하는 함수
def apply_mask_alpha(mask_rgba_img, alpha_factor):
    if mask_rgba_img is None:
        return None
    if mask_rgba_img.mode != 'RGBA':
        mask_rgba_img = mask_rgba_img.convert('RGBA')

    r, g, b, a = mask_rgba_img.split()
    a_np = np.array(a, dtype=np.float32)
    a_np = (a_np * alpha_factor).clip(0, 255).astype(np.uint8)
    new_a_pil = Image.fromarray(a_np)
    return Image.merge('RGBA', (r, g, b, new_a_pil))

# RGBA 마스크를 검은 배경의 RGB 이미지로 변환 (갤러리 표시용)
def convert_rgba_mask_to_rgb_on_black(rgba_mask_pil):
    if rgba_mask_pil is None:
        return None
    if rgba_mask_pil.mode == 'RGBA':
        black_bg = Image.new('RGB', rgba_mask_pil.size, (0,0,0))
        black_bg.paste(rgba_mask_pil, (0,0), rgba_mask_pil)
        return black_bg
    return rgba_mask_pil.convert('RGB') if rgba_mask_pil.mode != 'RGB' else rgba_mask_pil

# 범례 이미지 생성 함수 (입력: {클래스명: 색상} 딕셔너리)
def create_legend_image(name_to_color_map_for_legend, title="Legend"):
    item_height = 30
    swatch_size = 20
    text_x_padding_after_swatch = 10
    left_padding = 10
    right_padding = 10
    top_padding = 10
    bottom_padding = 10
    font_size = 14

    korean_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" # 나눔고딕 예시 경로
    font = None
    title_font = None

    try:
        font = ImageFont.truetype(korean_font_path, font_size)
        title_font = ImageFont.truetype(korean_font_path, int(font_size * 1.15))
    except IOError:
        print(f"지정된 한글 폰트 '{korean_font_path}'를 찾을 수 없습니다. 다른 폰트를 시도합니다.")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(font_size * 1.15))
        except IOError:
            print("DejaVuSans 폰트를 찾을 수 없습니다. Arial 또는 기본 폰트를 사용합니다.")
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                title_font = ImageFont.truetype("arialbd.ttf", int(font_size * 1.15))
            except IOError:
                font = ImageFont.load_default()
                title_font = ImageFont.load_default()

    num_items = len(name_to_color_map_for_legend)
    temp_img = Image.new("RGB", (1,1))
    temp_draw = ImageDraw.Draw(temp_img)

    title_bbox = temp_draw.textbbox((0,0), title, font=title_font) if title else (0,0,0,0)
    title_text_width = title_bbox[2] - title_bbox[0]
    title_text_height = title_bbox[3] - title_bbox[1]
    title_area_height = title_text_height + top_padding * 1.5 if title else top_padding

    max_item_name_width = 0
    if num_items > 0:
        for name in sorted(name_to_color_map_for_legend.keys()):
            bbox = temp_draw.textbbox((0,0), str(name), font=font)
            text_width = bbox[2] - bbox[0]
            if text_width > max_item_name_width:
                max_item_name_width = text_width

    img_width = int(left_padding + swatch_size + text_x_padding_after_swatch + max_item_name_width + right_padding)
    if title and title_text_width + left_padding + right_padding > img_width:
        img_width = int(title_text_width + left_padding + right_padding)
    img_width = max(img_width, 200)

    img_height = int(title_area_height + num_items * item_height + bottom_padding if num_items > 0 else title_area_height)
    img_height = max(img_height, 60)

    image = Image.new("RGB", (img_width, img_height), (240, 240, 240))
    draw = ImageDraw.Draw(image)

    current_y = top_padding
    if title:
        draw.text(
            (max(left_padding, (img_width - title_text_width) / 2), current_y),
            title, fill="black", font=title_font
        )
        current_y += title_text_height + top_padding * 0.5

    if num_items == 0 :
        no_item_text = "(선별된 클래스 없음)" # 문구 수정
        no_item_bbox = temp_draw.textbbox((0,0), no_item_text, font=font)
        no_item_width = no_item_bbox[2] - no_item_bbox[0]
        draw.text(
            (max(left_padding, (img_width - no_item_width) / 2), current_y),
            no_item_text, fill="grey", font=font
        )

    for class_name in sorted(name_to_color_map_for_legend.keys()):
        item_color = name_to_color_map_for_legend[class_name]
        item_center_y = current_y + item_height / 2

        swatch_y0 = item_center_y - swatch_size / 2
        swatch_x0 = left_padding
        draw.rectangle(
            [swatch_x0, swatch_y0, swatch_x0 + swatch_size, swatch_y0 + swatch_size],
            fill=item_color, outline="black"
        )
        
        text_bbox = temp_draw.textbbox((0,0), str(class_name), font=font)
        text_actual_height = text_bbox[3] - text_bbox[1] 
        text_y = item_center_y - text_actual_height / 2 - text_bbox[1] 

        draw.text(
            (swatch_x0 + swatch_size + text_x_padding_after_swatch, text_y),
            str(class_name), fill="black", font=font
        )
        current_y += item_height

    return image

# 메인 이미지 처리 함수
def process_images(image_files, alpha_slider_value):
    try:
        model_str_instance = YOLO(model_path_str)
        model_spa_instance = YOLO(model_path_spa)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        error_legend = create_legend_image({}, f"오류: 모델 로드 실패 ({e})")
        return [], [], [], error_legend

    # --- '공간' 또는 '구조'로 시작하는 클래스명만 필터링 및 색상맵 생성 ---
    spa_names_dict = model_spa_instance.names
    str_names_dict = model_str_instance.names

    all_class_names_to_keep = set()
    if spa_names_dict:
        for name in spa_names_dict.values():
            if name.startswith('공간') or name.startswith('구조'):
                all_class_names_to_keep.add(name)
    if str_names_dict:
        for name in str_names_dict.values():
            if name.startswith('공간') or name.startswith('구조'):
                all_class_names_to_keep.add(name)

    sorted_unique_class_names_to_keep = sorted(list(all_class_names_to_keep))

    combined_palette_list = generate_palette(len(sorted_unique_class_names_to_keep))
    master_class_name_to_color_map = {
        name: combined_palette_list[i] for i, name in enumerate(sorted_unique_class_names_to_keep)
    }

    # --- 통합 범례 이미지 생성 ---
    legend_title = "선별된 클래스 범례 ('공간' 또는 '구조' 시작)"
    combined_legend_img = create_legend_image(master_class_name_to_color_map, legend_title)
    # --- --- ---

    output_gallery_spa = []
    output_gallery_str = []
    output_gallery_final = []

    if not image_files:
        return [], [], [], combined_legend_img

    for image_file_obj in image_files:
        image_path = image_file_obj.name
        try:
            pil_image_original = Image.open(image_path).convert("RGB")
            img_array = np.array(pil_image_original)
        except Exception as e:
            print(f"이미지 파일 처리 중 오류 ({image_path}): {e}")
            continue

        # spa 예측
        result_spa_yolo = model_spa_instance(img_array, verbose=False)[0]
        mask_spa_rgba = generate_mask(result_spa_yolo,
                                      result_spa_yolo.names,
                                      master_class_name_to_color_map)

        # str 예측
        result_str_yolo = model_str_instance(img_array, verbose=False)[0]
        mask_str_rgba = generate_mask(result_str_yolo,
                                      result_str_yolo.names,
                                      master_class_name_to_color_map)

        final_overlayed_image_rgba = pil_image_original.convert('RGBA')
        if mask_spa_rgba is not None:
            mask_spa_with_user_alpha = apply_mask_alpha(mask_spa_rgba, alpha_slider_value)
            if mask_spa_with_user_alpha:
                 final_overlayed_image_rgba = Image.alpha_composite(final_overlayed_image_rgba, mask_spa_with_user_alpha)
        if mask_str_rgba is not None:
            mask_str_with_user_alpha = apply_mask_alpha(mask_str_rgba, alpha_slider_value)
            if mask_str_with_user_alpha:
                final_overlayed_image_rgba = Image.alpha_composite(final_overlayed_image_rgba, mask_str_with_user_alpha)

        final_overlayed_image_rgb = final_overlayed_image_rgba.convert('RGB')

        display_mask_spa_rgb = convert_rgba_mask_to_rgb_on_black(mask_spa_rgba)
        display_mask_str_rgb = convert_rgba_mask_to_rgb_on_black(mask_str_rgba)

        output_gallery_spa.append(display_mask_spa_rgb if display_mask_spa_rgb else pil_image_original)
        output_gallery_str.append(display_mask_str_rgb if display_mask_str_rgb else pil_image_original)
        output_gallery_final.append(final_overlayed_image_rgb)

    return output_gallery_spa, output_gallery_str, output_gallery_final, combined_legend_img

# Gradio UI
demo = gr.Interface(
    fn=process_images,
    inputs=[
        gr.File(file_count="multiple", label="도면 이미지 업로드 (.png, .jpg)", file_types=["image"]),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="오버레이 투명도 (0.0: 완전 투명, 1.0: 완전 불투명)")
    ],
    outputs=[
        gr.Gallery(label="SPA 마스크", columns=3, object_fit="contain", type="pil"),
        gr.Gallery(label="STR 마스크", columns=3, object_fit="contain", type="pil"),
        gr.Gallery(label="최종 오버레이 결과", columns=3, object_fit="contain", type="pil"),
        gr.Image(label="선별된 클래스 범례", type="pil", interactive=False)
    ],
    title="도면 Segmentation 웹 데모 (YOLOv11 기반)",
    description="여러 도면 이미지를 업로드하면, SPA(공간) 및 STR(구조) Segmentation을 수행하고 각각의 결과 및 오버레이를 보여줍니다."
)

if __name__ == "__main__":
    print("애플리케이션 시작 중...")
    demo.launch()