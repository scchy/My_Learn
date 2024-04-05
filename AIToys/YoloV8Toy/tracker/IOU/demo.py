'''
iou追踪示例
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

class IouTracker:
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

        # 追踪的IOU阈值
        self.sigma_iou = 0.5
        # detection threshold
        self.conf_thresh = 0.5
        # 颜色列表
        self.colors_list = self.getColorsList(len(self.objs_labels))

    def iou(sel,bbox1, bbox2):
        """
        计算两个bounding box的IOU
        @param bbox1: bounding box in format x1,y1,x2,y2
        @param bbox2: bounding box in format x1,y1,x2,y2

        @return: intersection-over-union (float)
        """

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # 计算重叠的矩形的坐标
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # 检查是否有重叠
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # 计算重叠矩形的面积以及两个矩形的面积
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        # 计算IOU
        return size_intersection / size_union


    def predict(self, frame):
        '''
        检测
        @param frame: 图片
        @return: 检测结果，格式：[{'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id},{'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id} ...]
        '''
        result = list(self.detection_model(frame, stream=True, conf=self.conf_thresh))[0]  # inference，如果stream=False，返回的是一个列表，如果stream=True，返回的是一个生成器
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array

        dets = [] # 检测结果
        # 参考：https://docs.ultralytics.com/modes/predict/#boxes
        # 遍历每个框
        for box in boxes.data:
            l,t,r,b = box[:4] # left, top, right, bottom
            conf, class_id = box[4:] # confidence, class
            # 排除不需要追踪的类别
            if class_id not in self.track_classes:
                continue
            # if class_id not in {3: 'car', 4: 'van', 5: 'truck', 8: 'bus'}:
            dets.append({'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id })
        return dets
    
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


        tracks_active = [] # 活跃的跟踪器
        frame_id = 1 # 帧ID
        track_idx = 1 # 跟踪器ID
        while True:
            # 读取一帧
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # 缩放至720p
                frame = cv2.resize(frame, (720, 1280))
                # 检测 [{'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id},{'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id} ...]
                dets = self.predict(frame)
                # 更新后的跟踪器
                updated_tracks = [] 
                # 遍历活跃的跟踪器
                for track in tracks_active:
                    if len(dets) > 0:
                        # 根据最大IOU更新跟踪器，先去explain.ipynb中看一下MAX用法
                        best_match = max(dets, key=lambda x: self.iou(track['bboxes'][-1], x['bbox'])) # 找出dets中与当前跟踪器（track['bboxes'][-1]）最匹配的检测框（IOU最大）
                        # 如果最大IOU大于阈值，则将本次检测结果加入跟踪器
                        if self.iou(track['bboxes'][-1], best_match['bbox']) > self.sigma_iou:
                            # 将本次检测结果加入跟踪器
                            track['bboxes'].append(best_match['bbox'])
                            track['max_score'] = max(track['max_score'], best_match['score'])
                            track['frame_ids'].append(frame_id)
                            # 更新跟踪器
                            updated_tracks.append(track)
                            # 删除已经匹配的检测框，避免后续重复匹配以及新建跟踪器
                            del dets[dets.index(best_match)]


                # 如有未分配的目标，创建新的跟踪器
                new_tracks = []
                for det in dets: # 未分配的目标，已经分配的目标已经从dets中删除
                    new_track = {
                        'bboxes': [det['bbox']], # 跟踪目标的矩形框
                        'max_score': det['score'], # 跟踪目标的最大score
                        'start_frame': frame_id,  # 目标出现的 帧id
                        'frame_ids': [frame_id],  # 目标出现的所有帧id
                        'track_id': track_idx,    # 跟踪标号
                        'class_id': det['class_id'], # 类别
                        'is_counted': False       # 是否已经计数
                    }
                    track_idx += 1
                    new_tracks.append(new_track)
                # 最终的跟踪器
                tracks_active = updated_tracks + new_tracks

                # 绘制跟踪器
                for track in tracks_active:
                    # 绘制跟踪器的矩形框
                    l,t,r,b = track['bboxes'][-1]
                    # 取整
                    l,t,r,b = int(l), int(t), int(r), int(b)
                    class_id = track['class_id']
                    cv2.rectangle(frame, (l,t), (r,b), self.colors_list[int(class_id)], 2)
                    # 绘制跟踪器的track_id + class_name + score（99.2%格式）
                    cv2.putText(frame, f"{track['track_id']}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    # cv2.putText(frame, f"{track['track_id']} {self.objs_labels[int(class_id)]} {track['max_score']:.2f}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    
                    
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
iou_tracker = IouTracker()
# 运行
iou_tracker.main()


