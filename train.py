import warnings
import os
from datetime import datetime
from ultralytics import YOLO

if __name__ == '__main__':

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model = YOLO(r"ultralytics/cfg/models/11/yolo11x_LSConv.yaml")
    model.train(data='ultralytics/cfg/datasets/MIDOG.yaml',
                cache=False,
                scale=0,
                degrees=180,
                mixup=0.3,
                close_mosaic=20,
                cls=1.2,
                imgsz=512,
                epochs=300,
                patience=40,
                batch=960,
                # workers=16,
                pretrained=False,
                save_period=10,
                single_cls=True,
                device=[0,1,2,3,4,5,6,7],
                optimizer='AdamW',
                amp=True,
                project='runs/train_single',
                name=f'exp_11X_LS3ema_1.2',  # 添加时间戳_LSConv
                )