# python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_single_lsgan \
#                                      --lambda-seg 0.0 \
#                                      --lambda-adv-target1 0.0 --lambda-adv-target2 0.01 \
#                                      --gan LS

# python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/GTA2Cityscapes_multi \
#                                      --lambda-seg 0.1 \
#                                      --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001 
#                                      --restore-from ./snapshots/GTA2Cityscapes_multi/GTA5_35000.pth


python train_gta2cityscapes_multi.py --snapshot-dir ./snapshots/train_paper_param \
                                     --lambda-seg 0.0 \
                                     --lambda-adv-target1 0.0 --lambda-adv-target2 0.01 \
                                     --gan LS
