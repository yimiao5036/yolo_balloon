import torch
import ultralytics
from train import predict_image

if __name__ == "__main__":
    predict_image('runs/detect/train/train-10/weights/best.pt', 'test.jpg')
    from ultralytics import YOLO

    # 加载你的模型
    model = YOLO('save_models/best.pt')

    # 导出模型
    model.export(format='onnx',opset=12)