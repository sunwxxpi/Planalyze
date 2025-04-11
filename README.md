# Planalyze



## Usage
### SPA datasets
SPA segmentation example by API
```
import os
from ultralytics import settings, YOLO
import torch

settings.update({'weights_dir': os.getcwd() + '/weights'})

model = YOLO('yolov8n-seg.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = model.train(
    data='SPA.yaml',
    epochs=10,
    imgsz=640,
    batch=8,  
    device=device,
    workers=0,
    task='segment'
)
```
