'''
botsort-bytetrack追踪示例
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time
import random
import os
# 命令行设置 export YOLO_VERBOSE=false 
model_weight = '/home/scc/sccWork/myGitHub/My_Learn/AIToys/YoloV8Toy/trainMyData/yoloTrain__VisDrone2019/n_100_8002/weights/best.pt'
media_file = '/home/scc/Downloads/AIToy/P2_Yolov8CarCount/5.tracking/1.iou_tracker/media/720p.mp4'


class Tracker:
    def __init__(self):

        # 加载检测模型
        self.detection_model = YOLO(model_weight)  
        # 获取类别 
        self.objs_labels = self.detection_model.names 
        # 打印类别
        # {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
        print(self.objs_labels)
        # 只处理car, van, truck, bus，即：轿车，货车，卡车，公交车
        self.track_classes = {3: 'car', 4: 'van', 5: 'truck', 8: 'bus'}
        

        # detection threshold
        self.conf_thresh = 0.5
        # 颜色列表
        self.colors_list = self.getColorsList(len(self.objs_labels))

    def getColorsList(self, num_colors):
        '''
        生成颜色列表
        '''
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # hex to bgr
        bgr_list = []
        for hex in hexs:
            bgr_list.append(tuple(int(hex[i:i+2], 16) for i in (4, 2, 0)))
        # 随机取num_colors个颜色
        final_list = [random.choice(bgr_list) for i in range(num_colors)]    
        return final_list    
    
    def main(self):
        '''
        主函数
        '''
        # 读取视频
        cap = cv2.VideoCapture(media_file)
  
        # 获取视频帧率、宽、高
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"fps: {fps}, width: {width}, height: {height}")


        while True:
            # 读取一帧
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # 缩放至720p
                frame = cv2.resize(frame, (720, 1280))

                # 检测+追踪 botsort.yaml | bytetrack.yaml
                result = list(
                    self.detection_model.track(
                        frame, 
                        persist=True, 
                        tracker='botsort.yaml', 
                        conf=self.conf_thresh
                    )
                )[0] 
                boxes = result.boxes  # Boxes object for bbox outputs
                boxes = boxes.cpu().numpy()  # convert to numpy array

                # 参考：https://docs.ultralytics.com/modes/predict/#boxes
                # 遍历每个框
                for box in boxes.data:
                    l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom
                    track_id, conf, class_id = box[4:]  # track_id, confidence, class
                    # 绘制框
                    cv2.rectangle(frame, (l,t), (r,b), self.colors_list[int(class_id)], 2)
                    # 绘制跟踪器的track_id + class_name + score（99.2%格式）
                    cv2.putText(frame, str(int(track_id)), (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # end time
                end_time = time.time()
                # FPS
                fps = 1 / (end_time - start_time)
                # 绘制FPS
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 显示
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            else:
                break


# 实例化
tracker = Tracker()
# 运行
tracker.main()

