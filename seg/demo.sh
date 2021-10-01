#!/bin/bash

# python -u ./tools/demo.py --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml TEST.TEST_MODEL_PATH /home/yoo/Downloads/deeplabv3_plus_resnet101_segmentron.pth --input_img /data/datasets/citys/leftImg8bit/test/bonn

# python -u ./tools/demo.py \
#     --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml \
#     --input-img /home/yoo/data/gta/images/ \
#     TEST.TEST_MODEL_PATH /home/yoo/workspace/SSL-Synthetic-Segmentation/seg/checkpoints/deeplabv3_plus_resnet101_segmentron.pth \
    
python -u ./tools/demo.py \
    --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml \
    --input-img /home/yoo/data/cityscapes/leftImg8bit/trainextra/ \
    TEST.TEST_MODEL_PATH /home/yoo/workspace/SSL-Synthetic-Segmentation/seg/checkpoints/deeplabv3_plus_resnet101_segmentron.pth \
    
    