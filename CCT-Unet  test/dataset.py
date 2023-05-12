# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from one_hot import mask_to_onehot


def make_dataset(data_path):
    samples = [ ]
    img_path = os.path.join(data_path, 'img')
    ann_path = os.path.join(data_path, 'ann')
    for img_name in os.listdir(ann_path):
        img = os.path.join(img_path, img_name)
        ann = os.path.join(ann_path, img_name)
        sample = (img, ann)
        samples.append(sample)
    return samples


class ProstateDataset_Test(Dataset):
    def __init__(self, root, transform=None, one_hot_mask=True):
        samples = make_dataset(root)

        self.palette = [ [ 0 ], [ 127 ], [ 255 ] ]
        self.samples = samples
        self.transform = transform
        self.one_hot_mask = one_hot_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, ann = self.samples[ index ]
        if self.one_hot_mask:
            ann_raw = Image.open(ann)
        else:
            ann_raw = Image.open(ann).convert('1')
        img = Image.open(img).convert('L')
        cor = img.size
        if self.transform:
            img, ann = self.transform(img, ann_raw)

        img = np.array(img)
        ann = np.array(ann)
        if self.one_hot_mask:
            ann = np.expand_dims(ann, axis=-1)
            ann = mask_to_onehot(ann, self.palette)
            ann = ann.transpose([2, 0, 1])
        else:
            ann = ann.astype(np.int)

        return img, ann, cor
