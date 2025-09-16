import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# runs/train_single/exp_20250812_174615/weights/epoch120.pt_LSbiema
if __name__ == '__main__':
    model = YOLO('/home/jupyter-x/ultralytics-main/runs/train_single/exp_11X_LS4ema_0.8/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/MIDOG.yaml',
              split='test',
              imgsz=512,
              batch=512,
              iou=0.3,
              conf=0.2,
              device=[0,],
              rect=False,
              save_json=True,
              project='runs/test',
              name='11X_LS4ema_0.8_0.2',
              plots=True,
              save_txt=True,
              save_conf=True,
              augment=True,
              visualize=True,
              )