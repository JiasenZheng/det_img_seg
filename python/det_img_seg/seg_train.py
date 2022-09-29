#!/usr/bin/python3

"""
Training class for instance segmentation using detectron2
"""

# common libraries
import os
# common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
# we use the COCO format for the dataset
from detectron2.data.datasets import register_coco_instances

class SegTrainer():
    """
    Custom trainer for instance segmentation
    """
    def __init__(self, model, weights, dataset_dir, output_dir, thing_classes):
        """
        Initialize the SegTrainer
        """
        # check inputs
        assert os.path.exists(model), "Model does not exist"
        assert os.path.exists(weights), "Weights do not exist"
        assert os.path.exists(dataset_dir), "Dataset directory does not exist"
        # set up the global variables
        self.model = model
        self.weights = weights
        self.dataset_dir = dataset_dir
        # set environment variables
        os.environ['DETECTRON2_DATASETS'] = self.dataset_dir
        self.output_dir = output_dir
        self.thing_classes = thing_classes
        # register the dataset
        register_coco_instances("seg_train", {}, os.path.join(self.dataset_dir, "train.json"), os.path.join(
            self.dataset_dir, "train"))
        # set thing classes
        MetadataCatalog.get("seg_train").set(thing_calsses=self.thing_classes)
        # get the metadata
        self.metadata = MetadataCatalog.get("seg_train")
        # initialize the config
        self.cfg = get_cfg()

    def init_cfg(self):
        """
        Initialize the config
        """
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model))
        self.cfg.DATASETS.TRAIN = ("seg_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model)
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 300
        self.cfg.SOLVER.STEPS = []
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
        self.cfg.OUTPUT_DIR = self.output_dir

    def train(self):
        """
        Train the model
        """
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

