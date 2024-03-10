# 加载模型

import torch
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamPredictor 
from fastsam import FastSAM, FastSAMPrompt


# 原始SAM模型
class SegmentModel(object):
    def __init__(self, model_path, model_type):
        self.model_path = model_path
        # 模型类型: vit_b, vit_l, vit_h
        if model_type is None:
            model_type = "_".join(model_path.split("_")[1:3])
        self.model_type = model_type
        # 模型设备: cuda, cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型
        self.sam_model = sam_model_registry[model_type](checkpoint=model_path)
        self.sam_model.to(self.device)
        # 创建预测器
        self.predictor = SamPredictor(self.sam_model)

        # 正在处理的image
        self.processing_image = None

    # 初始化image embedding
    def init_embedding(self, image_file):
        image = cv2.imread(image_file)
        self.processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 是否加载了embedding（只需要执行一次）
        self.predictor.set_image(self.processing_image)
        print("init embedding done.")

    # 推理
    def predict(self, points, labels):
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=False
        )
        return masks, scores


# FastSAM模型
class FastSamModel(object):
    def __init__(self, model_path):
        # 加载模型
        self.model = FastSAM(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prompt_process = None

    # 预处理，相当于init_embedding
    def preprocess(self, IMAGE_PATH):
        everything_results = self.model(
            IMAGE_PATH,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        self.prompt_process = FastSAMPrompt(
            IMAGE_PATH, everything_results, device=self.device
        )
        print("init done.")

    # 推理
    def predict(self, points, pointlabel):
        return self.prompt_process.point_prompt(points, pointlabel)
