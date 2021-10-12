LIST = './dataset/cityscapes_list/train.txt'
IMG_PATH = '/home/yoo/data/cityscapes/leftImg8bit/train/'
GT_PATH = '/home/yoo/data/cityscapes/gtFine/train/'
PLABEL_1 = '/home/yoo/workspace/SSL-Synthetic-Segmentation/Seg-Uncertainty/pseudo/aagc_640x360_b2_single_cutmix_real_bk/gtFine/train/'
PLABEL_2 = '/home/yoo/workspace/SSL-Synthetic-Segmentation/Seg-Uncertainty/pseudo/aagc_640x360_b2_single_cutmix_real/gtFine/train/'

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
matplotlib.use('TkAgg')
plt.style.use('seaborn-poster')

SAVE_PATH = '/home/yoo/share/paper/plabel/'

plt.figure(figsize=(16, 9))
def visualize(save=False, save_name='test', **images):
    """PLot images in one row."""
    n = len(images)
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    
    if save:
        plt.savefig(SAVE_PATH + save_name + '.png', bbox_inches='tight')
    # plt.show()
    # plt.draw()
    plt.pause(0.0001)

def get_file_lists(path, postfix='*.png'):
    return [y for x in os.walk(path) for y in glob(os.path.join(x[0], postfix))]

train_lists = open(LIST, 'r').read().splitlines()
img_lists = get_file_lists(IMG_PATH)
gt_lists = get_file_lists(GT_PATH, '*gtFine_color.png')
plabel_list1 = get_file_lists(PLABEL_1, '*leftImg8bit_color.png')
plabel_list2 = get_file_lists(PLABEL_2, '*leftImg8bit_color.png')

N = len(train_lists)

import random
l = random.sample(range(0, N), 100)

for _, idx in enumerate(l):
    visualize(
            save=True,
            save_name=str(idx),
            image=Image.open(img_lists[idx]),
            ground_truth=Image.open(gt_lists[idx]),
            psudo_label_1=Image.open(plabel_list1[idx]),
            psudo_label_2=Image.open(plabel_list2[idx]),    
    )
