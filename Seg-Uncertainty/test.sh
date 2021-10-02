#!/bin/sh

python evaluate_cityscapes.py \
    --restore-from ./snapshots/aagc_640x360_b1/GTA5_40000.pth
    # --restore-from ./snapshots/aagc_640x360_b1_student/GTA5_20000.pth
    
    
    # --restore-from ./snapshots/gc_640x360_b1/GTA5_220000.pth
    # --restore-from ./snapshots/GTA_TO_CITY_CO/GTA5_204000.pth
