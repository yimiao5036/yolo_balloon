import torch
from ultralytics import YOLO

# 1. 加载模型
model = YOLO("save_models/yolov8n-best.pt")

# 2. 创建一个模拟输入（batch_size=1, 3通道, 640x640）
#    根据模型训练时的输入尺寸调整，常用640
dummy_input = torch.randn(1, 3, 640, 640)

# 3. 进行推理（model.model 指向底层的 PyTorch 模型）
with torch.no_grad():
    # 对于YOLOv8检测模型，输出是一个列表
    output = model.model(dummy_input)[0]

# 4. 打印输出形状
print("模型输出形状:", output.shape)
# 你可能看到 torch.Size([1, 84, 8400]) 或 torch.Size([1, 8400, 84])