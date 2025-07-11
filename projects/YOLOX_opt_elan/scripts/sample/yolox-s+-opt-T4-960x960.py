#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# from https://drive.google.com/drive/folders/1A7-4oSJCtu_9lFQoAKZUkwxYAqGvzAny
import os

from yolox.exp import Exp_T4 as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.50
        self.input_size = (960, 960)
        self.test_size = (960, 960)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.multiscale_range = 0
        self.hsv_prob = 1.0

        self.spp = False
        self.stem = False
        self.act = "relu6"

        self.num_classes = 8
        self.elan = True
        self.eval_interval = 1
        self.warmup_epochs = 50
        self.no_aug_epochs = 50
        self.basic_lr_per_img = 0.001 / 64.0
        self.train_ann = "t4_dataset_v1_2_train_cleaned.json"
