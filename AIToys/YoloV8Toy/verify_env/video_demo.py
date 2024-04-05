# python3
from ultralytics import YOLO
import cv2 
import numpy as np 
import time 
import os


base_root = '/home/scc/Downloads/AIToy/P2_Yolov8CarCount/1.verify_env'
model = YOLO(os.path.join(base_root, 'yolov8n.pt'))
objs_labels = model.names  # get class labels
print(objs_labels)

# read vedio
cap = cv2.VideoCapture(os.path.join(base_root, 'test.mp4'))

while True:
    # read 1 frame
    st = time.time()
    ret, frame = cap.read()
    if ret:
        # detect
        res = list(model(frame, stream=True))[0]
        boxes = res.boxes.cpu().numpy()

        # reference: https://docs.ultralytics.com/modes/predict/#boxes
        for box in boxes.data:
            l, t, r, b = box[:4].astype(np.int32)
            conf, id_ = box[4:]
            # rbg
            cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
            # draw conf
            cv2.putText(
                frame, 
                f"{objs_labels[id_]} {conf*100:.1f}%",
                (l, t-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(0, 255, 0), 
                thickness=2
            )
        
        # ed 
        ed = time.time()
        fps = 1/(ed - st)
        cv2.putText(
            frame, f'FPS: {fps:.2f}', (10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(0, 255, 0), 
            thickness=2
        )
        # show 
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    
    else:
        break 

