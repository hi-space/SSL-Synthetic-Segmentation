#!/bin/sh

python train_seg.py  \
    --snapshot-dir ./snapshots/aagc_640x360_b2_single_cutmix_student_seg  \
    --drop 0.1  \
    --warm-up 5000  \
    --batch-size 2  \
    --learning-rate 2e-4  \
    --crop-size 640,360  \
    --lambda-seg 0.5   \
    --lambda-adv-target1 0.0002  \
    --lambda-adv-target2 0.001    \
    --lambda-me-target 0   \
    --lambda-kl-target 0.1   \
    --norm-style gn   \
    --class-balance   \
    --only-hard-label 80   \
    --max-value 7   \
    --gpu-ids 0,1   \
    --often-balance   \
    --use-se
    
    
    #  \
    # --autoaug \
    # --autoaug_target