#!/bin/sh

python evaluate_cityscapes.py \
    --restore-from ./snapshots/SE_GN_batchsize2_1024x512_pp_ms_me0_classbalance7_kl0.1_lr2_drop0.1_seg0.5/GTA5_25000.pth
    # --restore-from ./snapshots/GTA_TO_CITY_CO/GTA5_204000.pth
