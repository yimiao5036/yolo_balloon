import os
import uuid
import time
from pathlib import Path
import torch
from ultralytics import YOLO
import flask
import cv2
import numpy as np

app = flask.Flask(__name__)
model_path = 'save_models/yolo26n-best.pt'

# 确保保存目录存在
UPLOAD_DIR = Path("uploaded_images")
ANNOTATED_DIR = Path("annotated_images")
UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

# 全局加载模型
model = YOLO(model_path)


def save_image(img_bgr, save_dir, prefix="img"):
    """保存 BGR 格式的图片，返回保存路径"""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
    save_path = save_dir / filename
    cv2.imwrite(str(save_path), img_bgr)
    return str(save_path)


@app.route('/img_predict', methods=['POST'])
def img_predict():
    if 'image' not in flask.request.files:
        return flask.jsonify({'error': 'No image file provided'}), 400

    file = flask.request.files['image']
    if file.filename == '':
        return flask.jsonify({'error': 'Empty filename'}), 400

    # 读取图片为 OpenCV 格式
    try:
        img_bytes = file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image")
    except Exception as e:
        return flask.jsonify({'error': f'Failed to decode image: {str(e)}'}), 400

    # 1. 保存原始图片
    original_path = save_image(img, UPLOAD_DIR, "original")
    print(f"Original image saved to {original_path}")

    # 模型推理
    try:
        results = model(img, verbose=False)
        result = results[0]
    except Exception as e:
        return flask.jsonify({'error': f'Inference failed: {str(e)}'}), 500

    # 解析检测结果
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        for i in range(len(boxes)):
            detections.append({
                'bbox': boxes[i].tolist(),
                'confidence': float(confs[i]),
                'class_id': int(cls_ids[i]),
                'class_name': names[cls_ids[i]]
            })

    # 2. 绘制检测框并保存带标注的图片
    annotated_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    annotated_path = save_image(annotated_img, ANNOTATED_DIR, "annotated")
    print(f"Annotated image saved to {annotated_path}")

    # 可选：返回图片访问 URL（需要在 Flask 中配置静态路由或直接返回路径）
    # 假设你设置了 /static 目录映射，可返回 /static/uploaded_images/original_xxx.jpggit init
    return flask.jsonify({
        'success': True,
        'num_detections': len(detections),
        'detections': detections,
        'saved_original': original_path,
        'saved_annotated': annotated_path
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)