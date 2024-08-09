"""
Input: image_id, positive_pair_poi / positive_pair_geo
Purpose: get positive samples
Output: a pair of positive sample images
"""

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_id, pos_pair, image_dir, transform=None):
        self.image_id = np.load(image_id, allow_pickle=True)   # image_id.npy
        self.pos_pair = np.load(pos_pair, allow_pickle=True).reshape(1, -1)   # pos_pair_poi.npy / pos_pair_geo.npy
        self.image_dir = image_dir   # ./dataset/image/
        self.transform = transform   # data augmentation

    def __len__(self):
        return len(self.image_id)   # image num = 12200

    def __getitem__(self, idx):
        anchor_id = self.image_id[idx][0]
        anchor_image = Image.open(self.image_dir + anchor_id + '.png')   # anchor image

        pos_id = self.pos_pair[0, idx]
        pos_image = Image.open(self.image_dir + pos_id + '.png')  # positive image

        # resize -> 224 * 224, image to tensor
        if self.transform:
            anchor_sample = self.transform(anchor_image)   # anchor image tensor
            pos_sample = self.transform(pos_image)   # positive image tensor
        else:
            raise NotImplementedError

        return anchor_sample, pos_sample  # image tensor input model
