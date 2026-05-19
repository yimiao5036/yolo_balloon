import json
import os
from pathlib import Path


def convert_labelme_json_to_yolo(json_path, output_txt_path, img_width, img_height):
    """将单个 LabelMe JSON 转换为 YOLO txt 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取图片尺寸（如果 JSON 中没有，需要根据实际图片获取）
    # 优先使用 JSON 中的 imageWidth/Height，否则用传入的参数
    width = data.get('imageWidth', img_width)
    height = data.get('imageHeight', img_height)

    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        # 假设你的类别只有一个 'balloon'，映射为 id 0
        class_id = 0  # 如果有多个类别，可以建立一个字典映射

        points = shape['points']  # [[x1,y1], [x2,y2], ...] 多边形或矩形
        # 计算外接矩形（xmin, ymin, xmax, ymax）
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        # 计算中心点坐标和宽高（像素值）
        box_width = xmax - xmin
        box_height = ymax - ymin
        x_center = xmin + box_width / 2
        y_center = ymin + box_height / 2

        # 归一化
        x_center_norm = x_center / width
        y_center_norm = y_center / height
        width_norm = box_width / width
        height_norm = box_height / height

        # 写入一行
        yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

    # 写入 txt 文件
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))


def batch_convert(source_img_dir, source_json_dir, target_label_dir, target_img_dir=None):
    """
    批量转换
    source_img_dir: 存放原始图片的文件夹（例如 train/img）
    source_json_dir: 存放 labelme JSON 的文件夹（例如 train/label）
    target_label_dir: 转换后的 txt 输出文件夹（例如 train/labels）
    target_img_dir: 可选，用于复制图片到目标 images 文件夹（如果不提供，则只生成 txt）
    """
    os.makedirs(target_label_dir, exist_ok=True)
    if target_img_dir:
        os.makedirs(target_img_dir, exist_ok=True)

    # 获取所有图片文件（支持常见格式）
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    img_files = [f for f in os.listdir(source_img_dir) if Path(f).suffix.lower() in img_extensions]

    for img_file in img_files:
        stem = Path(img_file).stem  # 不带扩展名的文件名
        json_file = stem + '.json'
        json_path = os.path.join(source_json_dir, json_file)

        if not os.path.exists(json_path):
            print(f"警告：未找到 {json_file}，跳过图片 {img_file}")
            continue

        # 获取图片实际尺寸（用于归一化，如果 JSON 中没有 imageWidth/Height）
        from PIL import Image
        img_path_full = os.path.join(source_img_dir, img_file)
        try:
            with Image.open(img_path_full) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"无法读取图片 {img_file} 的尺寸：{e}")
            continue

        # 转换
        txt_output = os.path.join(target_label_dir, stem + '.txt')
        convert_labelme_json_to_yolo(json_path, txt_output, img_w, img_h)
        print(f"已转换：{img_file} -> {txt_output}")

        # 如果需要同步复制图片到目标 images 文件夹
        if target_img_dir:
            import shutil
            shutil.copy2(img_path_full, os.path.join(target_img_dir, img_file))


def main():
    # 根据你的实际路径修改
    base_dir = Path(r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\Balloon')

    # 训练集转换
    train_img_src = base_dir / 'train' / 'img'
    train_json_src = base_dir / 'train' / 'label'  # 你的 JSON 文件夹名叫什么？请确认
    train_label_dst = Path(r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\train\labels')
    train_img_dst = Path(r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\train\images')

    # 验证集转换
    val_img_src = base_dir / 'val' / 'img'
    val_json_src = base_dir / 'val' / 'label'
    val_label_dst = Path(r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\val\labels')
    val_img_dst = Path(r'D:\ProgramData\前端实习\代码记录\YOLOV8\dataset\val\images')

    print("转换训练集...")
    batch_convert(train_img_src, train_json_src, train_label_dst, train_img_dst)
    print("转换验证集...")
    batch_convert(val_img_src, val_json_src, val_label_dst, val_img_dst)


if __name__ == '__main__':
    main()