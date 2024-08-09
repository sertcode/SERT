"""
Input: id of images
Purpose: get positive samples of contrastive learning based on poi
Output: positive samples image id
"""

import numpy as np
import random


poi_num = np.load('dataset/poi_num.npy')
image_id = np.load('dataset/image_id.npy', allow_pickle=True)

pos_idx_list = []

for idx in range(poi_num.shape[0]):
    simi = np.linalg.norm(poi_num[idx, :] - poi_num, axis=1, keepdims=True)   # 2-Norm / Euclidean
    simi[idx, 0] = max(simi)
    most_simi = np.min(simi)
    most_simi_idx = np.where(simi == most_simi)
    pos_random = random.randint(0, len(most_simi_idx[0])-1)   # multiple most similar, randomly select one
    pos_idx = most_simi_idx[0][pos_random]
    pos_idx_list.append(image_id[pos_idx])

np.save('dataset/pos_pair_poi.npy', np.array(pos_idx_list))

