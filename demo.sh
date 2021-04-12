#!/bin/bash

# python -u ./tools/demo.py --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml TEST.TEST_MODEL_PATH /home/yoo/Downloads/deeplabv3_plus_resnet101_segmentron.pth --input_img /data/datasets/citys/leftImg8bit/test/bonn
python -u ./tools/demo.py --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml TEST.TEST_MODEL_PATH /home/yoo/workspace/SegmenTron/runs/checkpoints/DeepLabV3_Plus_resnet50_cityscape_2021-04-02-14-03/best_model.pth --input-img /data/datasets/citys/leftImg8bit/test/bonn/


