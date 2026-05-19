# train.py
from click.core import batch
from ultralytics import YOLO
import torch

def train_yolov8():
    # 1. 检查是否可用 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. 加载预训练模型（nano 版本，速度快；也可选 s, m, l, x）
    model = YOLO('yolo26n.pt')  # 自动下载预训练权重

    # 3. 开始训练
    results = model.train(
        data= r'dataset/data.yaml',
        epochs=100,               # 训练轮数
        imgsz=640,                # 输入图片尺寸
        batch=8,                 # 批次大小（根据显存调整）
        device=device,            # 自动选择 GPU 或 CPU
        workers=1,                # 数据加载线程数
        # lr0=0.0001,                 # 初始学习率
        momentum=0.937,           # SGD 动量
        weight_decay=0.0005,      # 权重衰减
        patience=150,              # 早停轮数（50轮没有提升就停止）
        save=True,                # 保存训练检查点
        save_period=10,           # 每10个epoch保存一次
        val=True,                 # 每个epoch结束时验证
        project='train',          # 结果保存根目录
        name='balloon_detector',  # 本次实验的子文件夹名
        exist_ok=True,            # 允许覆盖同名文件夹
        pretrained=True,          # 使用预训练权重
        optimizer='MuSGD',         # 自动选择优化器（SGD或AdamW）
        seed=42,                  # 随机种子，保证可复现
        verbose=True,             # 打印详细信息
        # hsv_h=0.02,
        # hsv_s=0.8,
        # hsv_v=0.5,
        # degrees=10,
        # translate=0.2,
        # scale=0.5,
        # shear=5,
        # fliplr=0.5,
        # mosaic=0.8,
        # mixup=0.2,
        # copy_paste=0.3,
    )

    # 4. 训练完成后，输出最佳模型路径
    print(f"Training completed. Best model saved at: {model.trainer.best}")
    return model

def validate_model(model_path, data_yaml):
    """在验证集上评估模型性能"""
    model = YOLO(model_path)
    torch.cuda.empty_cache()
    metrics = model.val(data=data_yaml,batch=4,workers=0)
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")
    return metrics

def predict_image(model_path, image_path):
    """使用训练好的模型对单张图片进行预测"""
    model = YOLO(model_path)
    results = model(image_path)
    results[0].show()  # 显示图片和检测框
    results[0].save(filename='prediction.jpg')  # 保存结果

if __name__ == '__main__':
    # 训练
    trained_model = train_yolov8()

    # # 验证最佳模型
    # validate_model('save_models/yolo26s-yolov8n-best.pt',
    #                r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\data.yaml')
