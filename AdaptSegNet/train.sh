python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_single_lsgan \
                                     --lambda-seg 0.0 \
                                     --lambda-adv-target1 0.0 --lambda-adv-target2 0.01 \
                                     --gan LS \
                                     --restore-from ./snapshots/GTA2Cityscapes_single_lsgan/GTA5_100000.pth