# run with docker

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# 加载模型
base_root = '/data'
model = YOLO(os.path.join(base_root, 'yolov8n.pt'))
objs_labels = model.names  # get class labels
print(objs_labels)


# 读取视频
cap = cv2.VideoCapture(os.path.join(base_root, 'test.mp4'))
# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
# writer
writer = cv2.VideoWriter("./test_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    # 读取一帧
    start_time = time.time()
    ret, frame = cap.read()
    if ret:
        # 检测
        result = list(model(frame, stream=True))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器

        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array
        
        # 参考：https://docs.ultralytics.com/modes/predict/#boxes
        # 遍历每个框
        for box in boxes.data:
            l,t,r,b = box[:4].astype(np.int32) # left, top, right, bottom
            conf, id = box[4:] # confidence, class
            # 绘制框
            cv2.rectangle(frame, (l,t), (r,b), (0,0,255), 2)
            # 绘制类别+置信度（格式：98.1%）
            cv2.putText(frame, f"{objs_labels[id]} {conf*100:.1f}%", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # end time
        end_time = time.time()
        # FPS
        fps = 1 / (end_time - start_time)
        # 绘制FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

        # 显示
        # # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        writer.write(frame)


    else:
        break