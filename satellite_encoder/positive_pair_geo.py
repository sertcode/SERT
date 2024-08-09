"""
Input: id of images
Purpose: get positive samples of contrastive learning based on id / position
Output: positive samples image id
"""

import numpy as np
import random


image_id = np.load('dataset/image_id.npy', allow_pickle=True)
image_id_split = np.array([row[0].split('_') for row in image_id], dtype='float64')   # X_Y -> X Y

pos_idx_list = []

for idx in range(image_id_split.shape[0]):
    simi = np.linalg.norm(image_id_split[idx, :] - image_id_split, axis=1, keepdims=True)   # 1-Norm / manhattan
    simi[idx, 0] = max(simi)
    most_simi = np.min(simi)
    most_simi_idx = np.where(simi == most_simi)
    pos_random = random.randint(0, len(most_simi_idx[0])-1)   # multiple most similar, randomly select one
    pos_idx = most_simi_idx[0][pos_random]
    pos_idx_list.append(image_id[pos_idx])

np.save('dataset/pos_pair_geo.npy', np.array(pos_idx_list))

