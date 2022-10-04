#!/usr/bin/python3

"""
Inference class for instance segmentation using detectron2
"""

import os, cv2, random
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model

class SegInference:
    """
    Inference class for instance segmentation using detectron2
    """
    def __init__(self, model, weights, image_dir, thing_classes):
        self.model = model
        self.weights = weights
        self.image_dir = image_dir
        self.thing_classes = thing_classes
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model))
        self.cfg.MODEL.WEIGHTS = os.path.join(self.weights, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
        self.predictor = DefaultPredictor(self.cfg)
        self.warmup_predictor()

    
    def warmup_predictor(self, num_warmup = 5):
        """
        Warm up the predictor
        """
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(num_warmup):
            self.predictor(img)

    def infer_once(self, img_path):
        """
        Run inference on the given image
        """
        # load the image
        img = cv2.imread(img_path)
        # run inference
        outputs = self.predictor(img)
        # get the predictions
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # show the image with time interval
        cv2.imshow("infer", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def infer_dir(self):
        """
        Run inference on the given directory
        """
        # get the list of images
        img_list = os.listdir(self.image_dir)
        # run inference on each image
        for img_name in img_list:
            img_path = os.path.join(self.image_dir, img_name)
            self.infer_once(img_path)

