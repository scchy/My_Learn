'''
botsort-bytetrack追踪示例
'''
from ultralytics import YOLO
import cv2
import numpy as np
import time
import random
import os
from shapely.geometry import Polygon, LineString
import json
# 命令行设置 export YOLO_VERBOSE=false 
model_weight = '/home/scc/sccWork/myGitHub/My_Learn/AIToys/YoloV8Toy/trainMyData/yoloTrain__VisDrone2019/n_100_8002/weights/best.pt'
media_file = '/home/scc/Downloads/AIToy/P2_Yolov8CarCount/5.tracking/1.iou_tracker/media/720p.mp4'
json_ = './proj2_snap.json'

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
        self.conf_thresh = 0.3
        # 颜色列表
        self.colors_list = self.getColorsList(len(self.objs_labels))
        # 读取标注的json文件
        self.area_json = self.read_labelme_json(json_)

    def read_labelme_json(self, json_file):
        '''
        读取labelme标注的json文件

        @param json_file: json文件路径
        @return: dict {'mask': mask, 'line': line}
        '''
        # 读取json文件
        with open(json_file, 'r') as f:
            area_json = json.load(f)
            shapes = area_json['shapes']

            mask_overlay = []
            line = []
            for shape in shapes:
                if shape['shape_type'] == 'line':
                    start_pt, end_pt = shape['points'][0], shape['points'][1]
                    line.append(start_pt)
                    line.append(end_pt)
                else:
                    mask_overlay.append(shape['points'])
            
            return {'mask': mask_overlay, 'line': line}

    def predict(self, frame):
        '''
        检测
        @param frame: 图片
        @return: 检测结果，格式：[{'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id}, ...]
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
            dets.append({'bbox': [l,t,r,b], 'score': conf, 'class_id': class_id })
        return dets
    
    def getColorsList(self, num_colors):
        '''
        生成颜色列表
        '''
        hexs = ('FF701F', 'FFB21D', 'CFD231', '48F90A', '520085', '3DDB86', '1A9334', '00D4BB', '00C2FF',
                '2C99A8', '344593', '6473FF', '0018EC', '8438FF', '520085', '92CC17','CB38FF', 'FF95C8', 'FF37C7', 'FF3838', 'FF9D97')
        # hex to bgr
        bgr_list = []
        for hex in hexs:
            bgr_list.append(tuple(int(hex[i:i+2], 16) for i in (4, 2, 0)))
        # 随机取num_colors个颜色
        # final_list = [random.choice(bgr_list) for i in range(num_colors)]    
        return bgr_list    
    
   
    def is_cross_line(self, line, bbox):
        x, y, w, h = bbox
        pt1 = (x, y)
        pt2 = (x + w, y)
        pt3 = (x + w, y + h)
        pt4 = (x, y + h)
        rectange = Polygon([pt1, pt2, pt3, pt4])
        path = LineString(line)
        return path.intersects(rectange)

    def apply_mask(self, raw_frame, raw_mask):
        # 遮挡区域坐标
        vertices_list = [np.array(n ,np.int32) for n in self.area_json['mask'] ]
        # 创建一个空白的mask，与原图大小一致
        mask = raw_mask.copy()
        # 每块区域填充白色
        for vertices in vertices_list:
            cv2.fillPoly(mask, [vertices], (255, 255, 255))

        # 使用mask覆盖原图，bitwise_not是取反操作，意思是将mask区域取反，即黑变白，白变黑
        # bitwise_and是与操作，即将原图中mask区域以外的区域置为0
        result = cv2.bitwise_and(raw_frame, cv2.bitwise_not(mask))

        return result
    
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
       
        count_results = {'up': 0, 'down': 0 }
        # 创建一个空白的mask，与原图大小一致
        raw_mask = np.zeros((height, width, 3), np.uint8)
        # 用字典记录tracker，格式：{id: {'xywh': c_bbox, 'is_counted': False}}
        pre_tracks = {}
        frame_index = 1 # 帧索引
        while True:
            # 读取一帧
            start_time = time.time()
            ret, raw_frame = cap.read()
            if ret:
                # 对原图进行遮挡处理
                frame = self.apply_mask(raw_frame, raw_mask)
                # raw_frame = frame
                # 缩放至720p
                # frame = cv2.resize(frame, (720, 1280))
                
                # 检测追踪
                result = list(self.detection_model.track(frame, persist=True, tracker='bytetrack.yaml', conf=self.conf_thresh))[0]  # botsort.yaml | bytetrack.yaml
                
                boxes = result.boxes  # Boxes object for bbox outputs
                boxes = boxes.cpu().numpy()  # convert to numpy array

                # 参考：https://docs.ultralytics.com/modes/predict/#boxes
                # 遍历每个框
                cross_line_color = (0,255,255) # 越界线的颜色  
                for box in boxes.data:
                    l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom
                    id, conf, class_id = box[4:]  # track_id, confidence, class
                    id = int(id)
                    # 绘制框
                    cv2.rectangle(raw_frame, (l,t), (r,b), self.colors_list[int(class_id)], 2)
                    # 绘制跟踪器的track_id + class_name + score（99.2%格式）
                    cv2.putText(raw_frame, str(id), (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                    c_bbox = [l, t, r-l, b-t] # convert xyxy to xywh
                    
                    if frame_index == 1:
                        # 第一帧不参与计数，因为无法判断方向
                        pre_tracks[id] = {'xywh': c_bbox, 'is_counted': False}
                        continue
                    
                    # 检查是否与计数线有交汇
                    if self.is_cross_line(self.area_json['line'], c_bbox):
                        # 与计数线有交汇了
                        if id not in pre_tracks:
                            # 如果目标第一次出现，则不参与计数
                            pre_tracks[id] = {'xywh': c_bbox, 'is_counted': False}
                            continue
                        # 如果已经被计数了，更新一下坐标，然后跳过
                        elif pre_tracks[id]['is_counted']:
                            pre_tracks[id]['xywh'] = c_bbox
                            continue
                        else:
                            # 如果没有被计数，则判断方向
                            # 获取上一帧的坐标
                            pre_c_bbox = pre_tracks[id]['xywh']
                            # 获取中心点坐标
                            pre_y = pre_c_bbox[1] + pre_c_bbox[3] // 2
                            # 获取本次坐标的中心点坐标
                            c_y = c_bbox[1] + c_bbox[3] // 2
                            # 如果本次坐标的中心点坐标大于上一帧的中心点坐标，则是向下的
                            if c_y > pre_y:
                                count_results['down'] += 1
                                cross_line_color = (255,0,255)
                            else:
                                count_results['up'] += 1
                                cross_line_color = (0,0,255)
                            # 更新一下坐标，然后标记为已经计数
                            pre_tracks[id]['is_counted'] = True
                            pre_tracks[id]['xywh'] = c_bbox

                # 设置半透明
                color = (0,0,0)
                alpha = 0.2
                l,t = 0,0
                r,b = l+240,t+120
                raw_frame[t:b,l:r,0] = raw_frame[t:b,l:r,0] * alpha + color[0] * (1-alpha)
                raw_frame[t:b,l:r,1] = raw_frame[t:b,l:r,1] * alpha + color[1] * (1-alpha)
                raw_frame[t:b,l:r,2] = raw_frame[t:b,l:r,2] * alpha + color[2] * (1-alpha)

                # end time
                end_time = time.time()
                # FPS
                fps = 1 / (end_time - start_time)
                # 绘制FPS
                cv2.putText(raw_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                
                # 绘制直线
                cv2.line(raw_frame, 
                         (int(self.area_json['line'][0][0]), int(self.area_json['line'][0][1])), 
                         (int(self.area_json['line'][1][0]), int(self.area_json['line'][1][1])), 
                         cross_line_color, 8)
                # 绘制up和down人数
                cv2.putText(raw_frame, f"up: {count_results['up']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(raw_frame, f"down: {count_results['down']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 显示
                cv2.imshow("frame", raw_frame)

                frame_index += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            else:
                break


# 实例化
tracker = Tracker()
# 运行
tracker.main()


