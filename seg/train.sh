#!/bin/bash

python -u tools/train.py --config-file configs/gta_deeplabv3_plus_resnet.yaml --resume ./runs/checkpoints/DeepLabV3_Plus_resnet101_gta_2021-09-22-05-49/1_6000.pth
#python -u tools/train.py --config-file configs/cityscapes_deeplabv3_plus_resnet.yaml
