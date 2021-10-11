#!/bin/sh

python eval_single.py \
   --restore-from ./snapshots/aagc_640x360_b2_single_cutmix_real_d1000/GTA5_30000.pth
   # --restore-from ./snapshots/aagc_640x360_b2_single_cutmix_real/GTA5_40000.pth
#    --restore-from ./snapshots/cityscapes_seg_d1000/GTA5_20000.pth
    # --restore-from ./snapshots/aagc_640x360_b2_d1000/GTA5_20000.pth
