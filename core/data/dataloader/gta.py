"""Prepare Cityscapes dataset"""
import os
import torch
import scipy.io as io
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset


class GTASegmentation(SegmentationDataset):
    BASE_DIR = 'gta'
    NUM_CLASS = 19

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
    ignore_index = 250
    class_map = dict(zip(valid_classes, range(19)))

    label_colours = dict(zip(range(NUM_CLASS), colors))

    def __init__(self, root='G:/data', split='train', mode=None, transform=None, **kwargs):
        super(GTASegmentation, self).__init__(root, split, mode, transform, **kwargs)
        self.root = os.path.join(root, self.BASE_DIR)
        print(self.root)
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/gta"
        self.images, self.mask_paths = _get_gta_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                              23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        maks = self._encode_segmap(np.array(mask, dtype=np.uint8))

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))
    
    def _encode_segmap(self, lbl):
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')


def _get_gta_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskpath = os.path.join(mask_folder, '', filename)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logging.info('cannot find the mask or image:', imgpath, maskpath)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    splits = io.loadmat(os.path.join(folder, 'split.mat'))
    img_folder = os.path.join(folder, 'images')
    label_folder = os.path.join(folder, 'labels')
    
    if split == 'train':
        ids = splits['trainIds'][:,0]
        
    elif split == 'val':
        ids = splits['valIds'][:200,0]
    elif split == 'test':
        ids = splits['testIds'][:,0]
    else:
        assert split == 'trainval'
        ids = np.concatenate((splits['trainIds'][:,0], splits['valIds'][:,0]))
    
    image_paths, mask_paths = get_path_pairs(img_folder, label_folder)
    return image_paths, mask_paths


if __name__ == '__main__':
    dataset = GTASegmentation()
