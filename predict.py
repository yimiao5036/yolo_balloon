import torch
import ultralytics
from train import predict_image

if __name__ == "__main__":
    predict_image('save_models/yolo26n-best.pt', 'test.jpg')
    from ultralytics import YOLO

    # 加载你的模型
    model = YOLO('save_models/yolo26n-best.pt')

    # 导出模型
    model.export(format='onnx',opset=12)