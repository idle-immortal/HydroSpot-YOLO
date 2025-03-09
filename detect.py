import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

if __name__ == '__main__':
    model = YOLO('F:/文件/研究生/水葫芦识别/ultralytics-20240822/ultralytics-main/runs/train/exp22/weights/best.pt') # select your model.pt path
    model.predict(source='F:/文件/研究生/水葫芦识别/ultralytics-20240822/ultralytics-main/dataset/cross_val/fold_0/images/train/picture_478.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='yolov8asf3',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )