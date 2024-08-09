"""
Input: image features (poi / geo)
Purpose1: train an attention MLP to fuse poi & geo features
Purpose2: train an MLP to evaluate feature performance
Output: MLP model parameters, prediction results
"""

import os
import numpy as np
import time
import warnings

import torch
from torch import nn

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

indicator = 'comment'   # comment / population / population_density   TODO: change indicator
eval_data_path = f'./dataset/eval/{indicator}.npy'
model_save_path = f'./result/model_checkpoint/prediction_mlp/{indicator}/'


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


# poi & geo features fusion
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

    def forward(self, x1, x2):
        # fuse poi & geo features
        features = torch.stack([x1, x2], dim=1)
        features = self.attention(features)

        # predict indicators
        res = self.linear1(features)
        res = self.act1(res)
        res = self.linear2(res)

        return res


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print('*** Load features ***')
    image_name = np.load('./dataset/image_id.npy', allow_pickle=True).tolist()
    image_name = [name for sublist in image_name for name in sublist]
    poi = np.load('./result/feature/poi_view_feature.npy')
    geo = np.load('./result/feature/geo_view_feature.npy')
    eval_data = np.load(eval_data_path, allow_pickle=True)   # indicator data

    image_id = eval_data[:, 0].tolist()
    image_value = eval_data[:, 1].tolist()

    image_value_log = [np.log(item + 1) for item in image_value]   # log transform
    feature_poi = np.zeros((len(image_id), poi.shape[1]))
    feature_geo = np.zeros((len(image_id), geo.shape[1]))

    # some images are missing ground truth, they are deleted from the evaluation
    for i in range(len(image_id)):
        tmp_im = image_id[i]
        img_idx = image_name.index(tmp_im)
        feature_poi[i, :] = poi[img_idx, :]
        feature_geo[i, :] = geo[img_idx, :]

    ground_truth = np.array(image_value_log)

    # divide train set, val set, test set
    print('*** Divide dataset ***')
    x = np.arange(0, feature_geo.shape[0])
    idx_train, idx_test, y_train, y_test = train_test_split(x, ground_truth, test_size=0.2, random_state=100)
    idx_train, idx_val, y_train, y_val = train_test_split(idx_train, y_train, test_size=0.25, random_state=100)

    x_train_poi = feature_poi[idx_train, :]
    x_train_geo = feature_geo[idx_train, :]
    x_train_poi = torch.as_tensor(x_train_poi, dtype=torch.float32).cuda()
    x_train_geo = torch.as_tensor(x_train_geo, dtype=torch.float32).cuda()
    y_train = torch.as_tensor(y_train.reshape((-1, 1)), dtype=torch.float32).cuda()

    x_val_poi = feature_poi[idx_val, :]
    x_val_geo = feature_geo[idx_val, :]
    x_val_poi = torch.as_tensor(x_val_poi, dtype=torch.float32).cuda()
    x_val_geo = torch.as_tensor(x_val_geo, dtype=torch.float32).cuda()
    y_val = torch.as_tensor(y_val.reshape((-1, 1)), dtype=torch.float32)

    x_test_poi = feature_poi[idx_test, :]
    x_test_geo = feature_geo[idx_test, :]
    x_test_poi = torch.as_tensor(x_test_poi, dtype=torch.float32).cuda()
    x_test_geo = torch.as_tensor(x_test_geo, dtype=torch.float32).cuda()
    y_test = torch.as_tensor(y_test.reshape((-1, 1)), dtype=torch.float32)

    learningRate = 0.005
    inputDim = feature_poi.shape[1]   # 512
    outputDim = 1
    epochs = 1200

    # train an MLP
    model = LinearRegression(inputDim, outputDim)
    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.01)

    print('*** Prediction head train ***')
    start_time = time.time()
    best_r2 = -float("inf")
    best_model = None
    for epoch in range(epochs):
        model.train()
        outputs = model(x_train_poi, x_train_geo)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train error
        with torch.no_grad():
            predicted_train = model(x_train_poi, x_train_geo).cpu()
            y_train_cpu = y_train.cpu()
            r2_train = r2_score(list(y_train_cpu), list(predicted_train))

        # val error
        with torch.no_grad():
            model.eval()
            predicted_eval = model(x_val_poi, x_val_geo).cpu()
            r2_val = r2_score(list(y_val), list(predicted_eval))

        # test error
        with torch.no_grad():
            model.eval()
            predicted_test = model(x_test_poi, x_test_geo).cpu().data.numpy()
            r2 = r2_score(list(y_test), list(predicted_test))
            rmse = np.sqrt(mean_squared_error(list(y_test), list(predicted_test)))
            mae = metrics.mean_absolute_error(list(y_test), list(predicted_test))

            # only save the min r2 model
            if r2 > best_r2:
                if best_model is not None:
                    os.remove(best_model)
                best_r2 = r2

                out = os.path.join(model_save_path, 'best_model.tar')
                torch.save(model.state_dict(), out)
                best_model = out

                print('----------------------------------------------------')
                print(f'Epoch = {epoch}')
                print(f'Loss = {loss}')
                # print(f'R2_train = {r2_train}')
                # print(f'R2_val = {r2_val}')
                print(f'R2 = {r2}')
                print(f'RMSE = {rmse}')
                print(f'MAE = {mae}')

    print('----------------------------------------------------')
    end_time = time.time()
    print(f'Time = {end_time - start_time}')
