import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def yolov26():
  model = YOLO('/root/.pyenv/runs/detect/drone_prediction/train-10/weights/last.pt')

  results = model.train(
      data='dataset/data.yaml',
      epochs = 150,
      imgsz = 640,
      batch =  96,
      device = '0',
      pretrained = True,
      optimizer = 'MuSGD',
      seed = 42,
      workers = 8,
      project='drone_prediction',
      resume = True
      )


def draw_fancy_boxes(img,results,conf_threshold,class_names=None,box_style='modern',alpha=0.3):
  """
  """
  boxes = results.boxes
  if boxes is  None:
    return img

  if class_names is None:
    class_names = results.names if hasattr(results,'names') else {0:'unknown'}

  colors = [
      (220,95,85),(100,150,80),(60,180,200),(220,200,70)
  ]

  overlay = img.copy()

  for i,box in enumerate(boxes):
    x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
    conf = float(box.conf[0])
    if conf < conf_threshold:
      continue
    cls_id = int(box.cls[0])
    cls_name = class_names[cls_id] if cls_id in class_names else str(cls_id)
    color = colors[cls_id % len(colors)]

    radius = 8
    mask = np.zeros((y2-y1,x2-x1,3),dtype=np.uint8)
    cv2.rectangle(mask, (0,0), (mask.shape[1], mask.shape[0]), (255,255,255),-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2,radius*2))
    mask = cv2.erode(mask, kernel)

    overlay[y1:y2,x1:x2] = cv2.bitwise_and(overlay[y1:y2,x1:x2],mask)
    overlay[y1:y2,x1:x2] = overlay[y1:y2, x1:x2]*(1-alpha) + np.array(color)*alpha

    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2, cv2.LINE_AA)

    label = f'{cls_name} {conf:.2f}'
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_x1 = x1
    label_y1 = y1 - th - 6
    label_x2 = x1 + tw + 8
    label_y2 = y1

    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(overlay, label, (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)

  result = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
  return result

def image_predict(model_path,image_path,conf=0.25,save_dir=None,show=True,figsize=(10,8),box_style='fancy'):
  """
  """
  model = YOLO(model_path)
  results = model(image_path,conf=conf)

  if save_dir:
    results.save(save_dir=save_dir)

  if box_style == 'fancy':
    annotated_img = draw_fancy_boxes(results[0].orig_img,results[0],conf_threshold=conf)
  else:
    annotated_img = results[0].plot()

  if show:
    rgb_img = cv2.cvtColor(annotated_img,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title(f'YOLO Detection (conf={conf})')
    plt.show()

  return results,annotated_img

def video_predict(model_path, video_path, output_path="result_video.mp4",live_preview=False,conf=0.25,box_style='fancy'):
  """
  """
  model = YOLO(model_path)
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError(f'无法打开视频文件: {video_path}')

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  frame_count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    results_frame = model(frame, conf=conf)
    if box_style == 'fancy':
      annotated_frame = draw_fancy_boxes(frame, results_frame[0], conf_threshold=conf)
    else:
      annotated_frame = results_frame[0].plot()

    out.write(annotated_frame)
    frame_count += 1

    if live_preview:
      cv2.imshow('Processing...', annotated_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  out.release()
  cv2.destroyAllWindows()

  print(f'✅视频处理完成，共处理 {frame_count} 帧，结果保存至: {output_path}')

  return output_path

if __name__ == '__main__':
  image_predict('save_models/yolo26s-best.pt','test.jpg',box_style='fancy')