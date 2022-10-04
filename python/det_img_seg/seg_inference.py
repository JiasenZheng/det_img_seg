#!/usr/bin/python3

"""
Inference class for instance segmentation using detectron2
"""

import os, cv2, random
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
