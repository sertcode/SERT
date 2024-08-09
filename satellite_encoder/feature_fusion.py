"""
Input: trained MLP model, poi-view feature, geo-view feature
Purpose: fuse poi & geo feature
Output: image feature
"""

import numpy as np
import torch
from torch import nn


city = 'dc'   # nyc / chi / dc   TODO: change city
model_path = './result/model_checkpoint/prediction_mlp/population/best_model.tar'
feature_save_path = f'./result/feature/3_city/{city}/{city}_feature.npy'


def weights_init_1(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)


def weights_init_2(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.l1 = torch.nn.Linear(in_size, hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = torch.nn.Linear(int(hidden_size), 1, bias=False)

        weights_init_2(self.l1)
        weights_init_1(self.l2)

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, input_size, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_size, output_size, bias=True)

        self.attention = Attention(in_size=input_size)
        weights_init_2(self.linear1)
        weights_init_2(self.linear2)

    def forward(self, x1, x2, return_attention=False):
        features = torch.stack([x1, x2], dim=1)
        attention_features = self.attention(features)

        if return_attention:
            return attention_features   # get Attention module result

        res = self.linear1(attention_features)
        res = self.act1(res)
        res = self.linear2(res)

        return res


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poi = np.load(f'./result/feature/3_city/{city}/{city}_poi.npy')
    geo = np.load(f'./result/feature/3_city/{city}/{city}_geo.npy')

    poi_feature = torch.as_tensor(poi, dtype=torch.float32).to(device)
    geo_feature = torch.as_tensor(geo, dtype=torch.float32).to(device)

    inputDim = poi_feature.shape[1]   # 512
    outputDim = 1

    print(f'*** Fusion {city} features ***')
    model = LinearRegression(inputDim, outputDim)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        image_features = model(poi_feature, geo_feature, return_attention=True)

    np.save(feature_save_path, image_features.cpu().numpy())



