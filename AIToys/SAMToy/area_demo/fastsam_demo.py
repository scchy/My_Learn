# -*- coding : utf-8 -*-
import cv2
import numpy as np
from sam_model import FastSamModel
import argparse

class AreaCalculator:
    def __init__(self, img_file):
        metric_cofig_file = "./config/metric_cofig.txt"
        metric_points = self.get_image_metric(metric_cofig_file)
        metric_length = 14  # 根据实际情况设置，单位是cm

        # 计算单位面积在图片中的像素个数
        self.area_per_pixel = self.calculate_area_per_pixel(metric_points, metric_length)
    
        # 加载模型
        model_path = "/home/scc/Downloads/AIToy/P3_Segment_Anything/3.area_demo/weights/weights/FastSAM-s.pt"
        self.model = FastSamModel(model_path=model_path)
        # 读取图片
        self.image_file = img_file
        self.origin_img_data = cv2.imread(img_file)
        self.image_data = self.origin_img_data.copy()

        # 初始化image embedding
        self.model.preprocess(img_file)

        self.segment_pts = []
        self.ignore_pts = []

    # 读取配置文件中的点
    def get_image_metric(self, metric_cofig_file): 
        # 文件每行一个点：x,y
        with open(metric_cofig_file, "r", encoding="utf8") as fr:
            lines = fr.readlines()
            points = []
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                x, y = line.split(",")
                points.append((int(x), int(y)))
            return points

    # 计算单位面积在图片中的像素个数
    def calculate_area_per_pixel(self, metric_points, metric_length):
        # 计算点之间的像素距离
        distance = np.linalg.norm(np.array(metric_points[0]) - np.array(metric_points[1]))
        print("distance is ", distance)
        # 计算单位面积在图片中的像素个数
        area_per_pixel = (distance * distance )/ (metric_length * metric_length)
        print("area_per_pixel is ", area_per_pixel)
        return area_per_pixel
    
    # 计算面积
    def calculate_area(self):
        
        input_point = np.array(self.segment_pts + self.ignore_pts) # 输入点
        input_label = np.array([1] * len(self.segment_pts) + [0] * len(self.ignore_pts)) # 输入点的标签
   
        maskes = self.model.predict(input_point, input_label) # 预测
        mask = maskes[0] # 获取mask
        # 获取mask中True的个数
        num_mask = mask.sum()
        area = num_mask / self.area_per_pixel  # 计算面积

        # 使用图片数组的方式贴图
        # 若mask为true，则将像素置为半透明
        alpha = 0.5
        color = (255,0,255)
        self.image_data[mask > 0,0] = self.image_data[mask > 0,0] * alpha + color[0] * (1-alpha)
        self.image_data[mask > 0,1] = self.image_data[mask > 0,1] * alpha + color[1] * (1-alpha)
        self.image_data[mask > 0,2] = self.image_data[mask > 0,2] * alpha + color[2] * (1-alpha)

        # 绘制点
        for pt in self.segment_pts:
            cv2.circle(self.image_data, pt, 10, (0, 255, 0), -1)
        for pt in self.ignore_pts:
            cv2.circle(self.image_data, pt, 10, (0, 0, 255), -1)

        # 绘制文字
        cv2.putText(
            self.image_data,
            f"Total Area is {area:.02f} cm^2",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 0),
            2,
        )
    # 鼠标事件
    def set_seg_point_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONUP: # 左键点击
            self.image_data = self.origin_img_data.copy()
            print("click L, " + str([x, y]))
            self.segment_pts.append((x, y))
            # 计算面积
            self.calculate_area()
        elif event == cv2.EVENT_RBUTTONUP: # 右键点击
            self.image_data = self.origin_img_data.copy()
            print("click R, " + str([x, y]))
            self.ignore_pts.append((x, y))
            # 计算面积
            self.calculate_area()

    # 运行
    def run(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.set_seg_point_event) # 设置鼠标事件
        while(True):
            cv2.imshow('image',self.image_data)
            # 退出
            if cv2.waitKey(20) & 0xFF == 27:
                break
            # 重置
            elif cv2.waitKey(20) & 0xFF == ord('r'):
                self.image_data = self.origin_img_data.copy()
                self.ignore_pts = []
                self.segment_pts = []
                
        cv2.destroyAllWindows()
       

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--img_file", type=str, default="./imgs/test.jpg")
args = parser.parse_args()
# init
area_app = AreaCalculator(args.img_file)
area_app.run()
