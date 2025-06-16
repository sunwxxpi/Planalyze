import os
import shutil
import tempfile
import numpy as np
import cv2
import gradio as gr
from datetime import datetime
from typing import Generator
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# ===================================================================
# 1. Configuration
# ===================================================================

# --- Model and Font Paths ---
MODEL_PATH_STRUCTURE = "/home/psw/Planalyze/runs/segment/20250421_034638/exp/weights/best.pt"
MODEL_PATH_SPATIAL = "./yolov8n-best-spa.pt"
KOREAN_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


# ===================================================================
# 2. Helper Functions
# ===================================================================

def generate_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """Generates a unique color palette for the given number of classes."""
    np.random.seed(42)
    return [
        tuple(np.random.randint(0, 256, size=3).tolist())
        for _ in range(num_classes)
    ]


def generate_mask(result: object, color_map: dict) -> Image.Image | None:
    """Creates an RGBA segmentation mask image from a YOLO result."""
    if result.masks is None or len(result.masks.data) == 0:
        return None

    h, w = result.orig_shape
    mask_img_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for i, mask_tensor in enumerate(result.masks.data):
        class_id = int(result.boxes.cls[i].item())
        class_name = result.names.get(class_id)
        if not class_name or class_name not in color_map:
            continue

        color = color_map[class_name]
        mask_np = mask_tensor.cpu().numpy()

        # Resize mask to the original image dimensions
        mask_resized = cv2.resize(
            mask_np, (w, h), interpolation=cv2.INTER_NEAREST
        )

        # Apply color and full opacity to the segmented pixels
        segment_pixels = mask_resized > 0.5
        mask_img_rgba[segment_pixels, :3] = color
        mask_img_rgba[segment_pixels, 3] = 255  # Alpha: Opaque

    if np.any(mask_img_rgba[:, :, 3] > 0):
        return Image.fromarray(mask_img_rgba, 'RGBA')
    return None


def apply_mask_alpha(
    mask_rgba_img: Image.Image, alpha_factor: float
) -> Image.Image | None:
    """Adjusts the transparency of an RGBA mask image."""
    if mask_rgba_img is None:
        return None

    r, g, b, a = mask_rgba_img.convert('RGBA').split()
    a_np = (np.array(a, dtype=np.float32) * alpha_factor)
    a_np = a_np.clip(0, 255).astype(np.uint8)
    return Image.merge('RGBA', (r, g, b, Image.fromarray(a_np)))


def convert_rgba_to_rgb_on_black(
    rgba_mask_pil: Image.Image,
) -> Image.Image | None:
    """Converts an RGBA mask to RGB with a black background for display."""
    if rgba_mask_pil is None:
        return None
    if rgba_mask_pil.mode == 'RGBA':
        black_bg = Image.new('RGB', rgba_mask_pil.size, (0, 0, 0))
        black_bg.paste(rgba_mask_pil, (0, 0), rgba_mask_pil)
        return black_bg
    return rgba_mask_pil.convert('RGB')


def create_legend_image(
    name_to_color_map: dict, title: str = "Legend"
) -> Image.Image:
    """Creates a legend image from class names and their assigned colors."""
    item_height, swatch_size, font_size = 24, 18, 14
    padding = 10

    try:
        font = ImageFont.truetype(KOREAN_FONT_PATH, font_size)
        title_font = ImageFont.truetype(KOREAN_FONT_PATH, int(font_size * 1.2))
    except IOError:
        print("Warning: Korean font not found. Using default font.")
        font = title_font = ImageFont.load_default()

    # Dynamically calculate image width based on text length
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = max(
        (temp_draw.textbbox((0, 0), str(name), font=font)[2]
         for name in name_to_color_map),
        default=0,
    )
    title_w = temp_draw.textbbox((0, 0), title, font=title_font)[2]
    img_w = max(200, padding * 3 + swatch_size + max_text_w,
                padding * 2 + title_w)
    img_h = padding * 2 + 24 + (len(name_to_color_map) * item_height)

    image = Image.new("RGB", (img_w, img_h), (240, 240, 240))
    draw = ImageDraw.Draw(image)

    draw.text(((img_w - title_w) / 2, padding), title, fill="black",
              font=title_font)

    current_y = padding + 24
    if not name_to_color_map:
        no_item_text = "(No classes detected)"
        w, h = temp_draw.textbbox((0, 0), no_item_text, font=font)[2:4]
        draw.text(((img_w - w) / 2, current_y), no_item_text, fill="grey",
                  font=font)
    else:
        for name, color in sorted(name_to_color_map.items()):
            swatch_y = current_y + (item_height - swatch_size) / 2
            draw.rectangle(
                [padding, swatch_y, padding + swatch_size, swatch_y + swatch_size],
                fill=color,
                outline="black",
            )
            draw.text(
                (padding + swatch_size + 10, current_y + 2),
                name,
                fill="black",
                font=font,
            )
            current_y += item_height

    return image


# ===================================================================
# 3. Core Logic
# ===================================================================

def setup_models_and_legend() -> tuple[object, object, dict, Image.Image]:
    """Loads YOLO models and prepares a unified class legend on app start."""
    print("Step 1: Loading models and generating legend...")
    try:
        if not all(map(os.path.exists,
                       [MODEL_PATH_SPATIAL, MODEL_PATH_STRUCTURE])):
            raise FileNotFoundError("One or both model files not found.")
        model_spa = YOLO(MODEL_PATH_SPATIAL)
        model_str = YOLO(MODEL_PATH_STRUCTURE)
    except Exception as e:
        print(f"Error loading models: {e}")
        error_msg = f"Error: Failed to load models.\nDetails: {e}"
        return None, None, None, create_legend_image({}, title=error_msg)

    # Create a unified list of unique class names from both models
    all_names = list(model_spa.names.values()) + list(model_str.names.values())
    class_names = sorted(list({
        name for name in all_names
        if name and (name.startswith('공간') or name.startswith('구조'))
    }))

    palette = generate_palette(len(class_names))
    color_map = {name: palette[i] for i, name in enumerate(class_names)}
    legend_img = create_legend_image(color_map, "Class Legend")

    print("Step 1 Complete: Models and legend are ready.")
    return model_spa, model_str, color_map, legend_img


def run_inference_and_generate_masks(
    image_files: list, model_spa: object, model_str: object, color_map: dict
) -> tuple:
    """Runs inference on uploaded images and creates masks for each model."""
    print("Step 2: Starting image inference and mask generation...")
    if not image_files or not all([model_spa, model_str]):
        return ([],) * 10

    originals, spa_masks_rgb, str_masks_rgb = [], [], []
    state_originals, state_spa_masks_rgba, state_str_masks_rgba = [], [], []
    filenames = []

    for image_file in image_files:
        filename = os.path.basename(image_file.name)
        filenames.append(filename)

        pil_image = Image.open(image_file.name).convert("RGB")
        img_array = np.array(pil_image)

        originals.append((pil_image, filename))
        state_originals.append(pil_image)

        # Process with the spatial (SPA) model
        res_spa = model_spa(img_array, verbose=False)[0]
        mask_spa_rgba = generate_mask(res_spa, color_map)
        state_spa_masks_rgba.append(mask_spa_rgba)
        spa_masks_rgb.append(
            (convert_rgba_to_rgb_on_black(mask_spa_rgba) or pil_image, filename)
        )

        # Process with the structure (STR) model
        res_str = model_str(img_array, verbose=False)[0]
        mask_str_rgba = generate_mask(res_str, color_map)
        state_str_masks_rgba.append(mask_str_rgba)
        str_masks_rgb.append(
            (convert_rgba_to_rgb_on_black(mask_str_rgba) or pil_image, filename)
        )

    print(f"Step 2 Complete: Processed {len(image_files)} images.")
    return (
        originals, spa_masks_rgb, str_masks_rgb,
        originals, spa_masks_rgb, str_masks_rgb,
        state_originals, state_spa_masks_rgba, state_str_masks_rgba, filenames
    )


def _create_final_overlay_images(
    alpha: float,
    original_images: list[Image.Image],
    spa_masks: list[Image.Image],
    str_masks: list[Image.Image],
    filenames: list[str]
) -> tuple[list, list]:
    """Overlays spatial and structural masks onto the original images."""
    if not original_images:
        return [], []

    final_gallery_imgs, final_state_imgs = [], []

    for i, orig_img in enumerate(original_images):
        final_img_rgba = orig_img.convert('RGBA')

        # Composite the adjusted masks sequentially onto the image
        if spa_masks and i < len(spa_masks) and spa_masks[i]:
            spa_mask_adj = apply_mask_alpha(spa_masks[i], alpha)
            final_img_rgba = Image.alpha_composite(final_img_rgba, spa_mask_adj)
        if str_masks and i < len(str_masks) and str_masks[i]:
            str_mask_adj = apply_mask_alpha(str_masks[i], alpha)
            final_img_rgba = Image.alpha_composite(final_img_rgba, str_mask_adj)

        final_rgb = final_img_rgba.convert('RGB')
        final_gallery_imgs.append((final_rgb, filenames[i]))
        final_state_imgs.append((final_rgb, filenames[i]))

    return final_gallery_imgs, final_state_imgs


def save_results_as_zip(
    originals: list, spas: list, strs: list, finals: list,
    filenames: list[str], legend_img: Image.Image
) -> Generator[tuple[gr.Button, gr.File], None, None]:
    """Saves all analysis results into a structured ZIP file."""
    yield gr.Button(value="Saving...", interactive=False), gr.File(visible=False)
    if not originals:
        yield gr.Button(value="Save Results", interactive=True), gr.File(visible=False)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = os.path.join(temp_dir, "floor_plan_analysis_results")
        dir_paths = {
            "original": os.path.join(base_path, "1_original"),
            "spa_masks": os.path.join(base_path, "2_spa_masks"),
            "str_masks": os.path.join(base_path, "3_str_masks"),
            "final": os.path.join(base_path, "4_final_overlay"),
        }
        for path in dir_paths.values():
            os.makedirs(path)

        # Save each image type to its respective directory
        image_data_map = {
            "original": (originals, "og"),
            "spa_masks": (spas, "spa"),
            "str_masks": (strs, "str"),
            "final": (finals, "overlay"),
        }
        for key, (img_list, suffix) in image_data_map.items():
            for i, (img, _) in enumerate(img_list):
                base_name = os.path.splitext(filenames[i])[0]
                img.save(os.path.join(dir_paths[key], f"{base_name}_{suffix}.png"))

        if legend_img:
            legend_img.save(os.path.join(base_path, "class_legend.png"))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_base_name = os.path.join(
            tempfile.gettempdir(), f"analysis_results_{timestamp}"
        )
        zip_filepath = shutil.make_archive(zip_base_name, 'zip', base_path)

    yield gr.Button(value="Save Results", interactive=True), gr.File(
        value=zip_filepath, visible=True
    )


# ===================================================================
# 4. Gradio UI and Event Handlers
# ===================================================================

THEME = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Pretendard"), "ui-sans-serif", "system-ui"],
)
CSS_CODE = "footer { display: none !important; }"

with gr.Blocks(theme=THEME, title="Floor Plan Analyzer", css=CSS_CODE) as demo:
    # ---------------------------------------------------------------
    # 4.1. State Components (for non-visible data storage)
    # ---------------------------------------------------------------
    model_spa_state = gr.State(None)
    model_str_state = gr.State(None)
    color_map_state = gr.State(None)
    legend_image_state = gr.State(None)

    # States for image processing pipeline
    original_images_state = gr.State([])  # Original PIL images
    raw_spa_masks_state = gr.State([])    # RGBA spatial masks
    raw_str_masks_state = gr.State([])    # RGBA structural masks
    original_filenames_state = gr.State([])

    # States for referencing gallery data for ZIP saving
    original_gallery_state = gr.State([])
    spa_gallery_state = gr.State([])
    str_gallery_state = gr.State([])
    final_gallery_state = gr.State([])

    # ---------------------------------------------------------------
    # 4.2. UI Layout
    # ---------------------------------------------------------------
    gr.Markdown("# Floor Plan Segmentation Analysis (Powered by YOLO)")
    gr.Markdown(
        'Upload a floor plan image to analyze the "SPA (space)" and '
        '"STR (structure)" classes. The results are displayed overlaid '
        'on the original image.'
    )

    with gr.Row():
        with gr.Column(scale=3, min_width=450):
            image_upload = gr.File(
                file_count="multiple",
                label="Upload Floor Plan Images (.png, .jpg)",
                file_types=["image"],
            )
            alpha_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                label="Overlay Opacity",
                info="Adjusts the transparency of the overlay mask. "
                     "(0.0 = fully transparent ~ 1.0 = fully opaque)",
            )
            with gr.Row():
                run_button = gr.Button("Run Analysis", variant="primary")
                apply_opacity_button = gr.Button(
                    "Apply Opacity", visible=False
                )
                reset_button = gr.Button("Reset", visible=False)

            save_button = gr.Button("Save Results", visible=False)
            download_link = gr.File(
                label="Download Results", visible=False
            )

        with gr.Column(scale=2, min_width=300):
            legend_image = gr.Image(
                label="Class Legend", type="pil", interactive=False, height=420
            )

    with gr.Tabs(elem_id="output_tabs") as output_tabs:
        with gr.TabItem("Original Floor Plan"):
            original_gallery = gr.Gallery(
                label="Original Images", columns=3, object_fit="contain",
                type="pil", height="auto"
            )
        with gr.TabItem("SPA Masks (Space)"):
            spa_gallery = gr.Gallery(
                label="SPA Masks", columns=3, object_fit="contain",
                type="pil", height="auto"
            )
        with gr.TabItem("STR Masks (Structure)"):
            str_gallery = gr.Gallery(
                label="STR Masks", columns=3, object_fit="contain",
                type="pil", height="auto"
            )
        with gr.TabItem("Final Overlay Results", id="final_tab"):
            final_gallery = gr.Gallery(
                label="Final Overlay Results", columns=3,
                object_fit="contain", type="pil", height="auto"
            )

    # ---------------------------------------------------------------
    # 4.3. Event Handler Functions and Connections
    # ---------------------------------------------------------------
    def apply_opacity_and_update_ui(
        alpha, orig_imgs, spa_masks, str_masks, fnames
    ):
        """Handler to update the final gallery when the opacity slider changes."""
        print(f"Adjusting opacity to {alpha}...")
        yield (gr.Button(value="Applying...", interactive=False),
               None, None)

        final_gallery_imgs, final_state_imgs = _create_final_overlay_images(
            alpha, orig_imgs, spa_masks, str_masks, fnames
        )

        print("Opacity adjustment complete.")
        yield (gr.Button(value="Apply Opacity", interactive=True),
               final_gallery_imgs, final_state_imgs)

    def reset_analysis_data():
        """Resets the UI to its initial state, clearing all image data."""
        print("Resetting UI to initial state...")
        state_resets = ([],) * 8
        component_resets = (
            gr.File(value=None, visible=True),    # image_upload
            None, None, None, None,               # 4 Galleries
            gr.File(value=None, visible=False),   # download_link
            gr.Button(visible=True),              # run_button
            gr.Button(visible=False),             # apply_opacity_button
            gr.Button(visible=False),             # reset_button
            gr.Button(visible=False),             # save_button
        )
        return state_resets + component_resets

    # Event 1: Load models and legend when the app starts.
    def on_load():
        model_spa, model_str, color_map, legend = setup_models_and_legend()
        return model_spa, model_str, color_map, legend, legend

    demo.load(
        fn=on_load,
        inputs=None,
        outputs=[
            model_spa_state,
            model_str_state,
            color_map_state,
            legend_image,
            legend_image_state,
        ],
    )

    # Event 2: Run analysis when the 'Run Analysis' button is clicked.
    run_click_event = run_button.click(
        fn=run_inference_and_generate_masks,
        inputs=[
            image_upload,
            model_spa_state,
            model_str_state,
            color_map_state,
        ],
        outputs=[
            original_gallery, spa_gallery, str_gallery,
            original_gallery_state, spa_gallery_state, str_gallery_state,
            original_images_state, raw_spa_masks_state,
            raw_str_masks_state, original_filenames_state,
        ],
        show_progress="full",
    )
    # Chain events: After inference, create final overlays and update UI visibility.
    run_click_event.then(
        fn=_create_final_overlay_images,
        inputs=[
            alpha_slider,
            original_images_state,
            raw_spa_masks_state,
            raw_str_masks_state,
            original_filenames_state,
        ],
        outputs=[final_gallery, final_gallery_state],
        show_progress="full",
    ).then(
        fn=lambda: (
            gr.Tabs(selected="final_tab"),
            gr.Button(visible=False),  # run_button
            gr.File(visible=False),    # image_upload
            gr.Button(visible=True),   # apply_opacity_button
            gr.Button(visible=True),   # reset_button
            gr.Button(visible=True)    # save_button
        ),
        inputs=None,
        outputs=[
            output_tabs, run_button, image_upload,
            apply_opacity_button, reset_button, save_button,
        ],
    )

    # Event 3: Re-apply opacity when the 'Apply Opacity' button is clicked.
    apply_opacity_button.click(
        fn=apply_opacity_and_update_ui,
        inputs=[
            alpha_slider,
            original_images_state,
            raw_spa_masks_state,
            raw_str_masks_state,
            original_filenames_state,
        ],
        outputs=[apply_opacity_button, final_gallery, final_gallery_state],
    )

    # Event 4: Reset the UI when the 'Reset' button is clicked.
    reset_button.click(
        fn=reset_analysis_data,
        inputs=None,
        outputs=[
            original_images_state, raw_spa_masks_state, raw_str_masks_state,
            original_filenames_state, original_gallery_state,
            spa_gallery_state, str_gallery_state, final_gallery_state,
            image_upload, original_gallery, spa_gallery, str_gallery,
            final_gallery, download_link, run_button, apply_opacity_button,
            reset_button, save_button,
        ],
        cancels=[run_click_event],
    )

    # Event 5: Save results as a ZIP file.
    save_button.click(
        fn=save_results_as_zip,
        inputs=[
            original_gallery_state,
            spa_gallery_state,
            str_gallery_state,
            final_gallery_state,
            original_filenames_state,
            legend_image_state,
        ],
        outputs=[save_button, download_link],
        show_progress="full",
    )

# ===================================================================
# 5. Application Launch
# ===================================================================
if __name__ == "__main__":
    print("Starting Gradio application...")
    demo.launch()