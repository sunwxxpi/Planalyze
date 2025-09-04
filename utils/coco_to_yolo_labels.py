from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='./dataset/Training/labels/TL_SPA', use_segments=True, cls91to80=False)