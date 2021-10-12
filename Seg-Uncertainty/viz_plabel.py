LIST = './dataset/cityscapes_list/train.txt'
IMG_PATH = '/home/yoo/data/cityscapes/leftImg8bit/train/'
GT_PATH = '/home/yoo/data/cityscapes/gtFine/train/'
PLABEL_1 = '/home/yoo/workspace/SSL-Synthetic-Segmentation/Seg-Uncertainty/pseudo/aagc_640x360_b2_single_cutmix_real_bk/gtFine/train/'
PLABEL_2 = '/home/yoo/workspace/SSL-Synthetic-Segmentation/Seg-Uncertainty/pseudo/aagc_640x360_b2_single_cutmix_real/gtFine/train/'

IMG_POSTFIX = '_leftImg8bit.png'
GT_POSTFIX = '_gtFine_color.png'
PLABEL_POSTFIX = '_leftImg8bit_color.png'

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
import random
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
    else:
        plt.show()
        plt.pause(0.0001)

def get_file_lists(path, postfix='*.png'):
    return [{os.path.basename(y) : y} for x in os.walk(path) for y in glob(os.path.join(x[0], postfix))]

def search_key(key, item_lists):
    return [item for item in item_lists if key in item][0]

train_lists = open(LIST, 'r').read().splitlines()
img_lists = get_file_lists(IMG_PATH)
gt_lists = get_file_lists(GT_PATH, '*' + GT_POSTFIX)
plabel_list1 = get_file_lists(PLABEL_1, '*' + PLABEL_POSTFIX)
plabel_list2 = get_file_lists(PLABEL_2, '*' + PLABEL_POSTFIX)

N = len(train_lists)
l = random.sample(range(0, N), 100)

for _, idx in enumerate(l):
    key = os.path.basename(train_lists[idx]).rsplit('_', 1)[0]
    print(key)

    # find img_lists
    dict_img = search_key(key + IMG_POSTFIX, img_lists)
    img = dict_img[list(dict_img)[0]]

    # find gt_lists
    dict_gt = search_key(key + GT_POSTFIX, gt_lists)
    gt = dict_gt[list(dict_gt)[0]]

    # find plabel_lists
    dict_plabel1 = search_key(key + PLABEL_POSTFIX, plabel_list1)
    plabel1 = dict_plabel1[list(dict_plabel1)[0]]

    # find plabel_lists2
    dict_plabel2 = search_key(key + PLABEL_POSTFIX, plabel_list2)
    plabel2 = dict_plabel2[list(dict_plabel2)[0]]

    visualize(
        save=True,
        save_name=key,
        image=Image.open(img),
        ground_truth=Image.open(gt),
        psudo_label_1=Image.open(plabel1),
        psudo_label_2=Image.open(plabel2),
    )