from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='./drawing_data/Validation/label/VL_STR', use_segments=True, cls91to80=False)