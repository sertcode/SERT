from model import Sert
from utils import *
import argparse
import random
import os
import time
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# parameters
parser = argparse.ArgumentParser()
parser.add_argument('--source_city', type=str, default='NY', help='NY, CHI')
parser.add_argument('--target_city', type=str, default='DC')
parser.add_argument('--travel_mode', type=str, default='Taxi', help='Taxi, Bike')
parser.add_argument('--data_type', type=str, default='pickup', help='pickup, dropoff')
parser.add_argument('--data_amount', type=int, default=30, help='30, 7, 3')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--transfer_weight', type=float, default=0.4, help='transfer loss weight')
parser.add_argument('--temperature', type=float, default=0.6, help='contrastive learning temperature parameter')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--max_norm', type=float, default=3, help='clip_grad_norm parameter')
args = parser.parse_args()


# random seed
if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Use device: {device}.')
print(f'Run SERT, from [{args.source_city} to {args.target_city}], use [{args.travel_mode} {args.data_type}] data,'
      f' with [{args.data_amount} days] target data.')


# load source city data, shape = (8784, lng, lat), 8784 = 366 * 24
source_data = np.load(f'./data/{args.source_city}/{args.travel_mode}{args.source_city}_{args.data_type}.npy')
source_lng, source_lat = source_data.shape[1], source_data.shape[2]
source_mask = source_data.sum(0) > 0   # overlook the regions with no traffic data
# 0/1 mask tensor, shape = (1, lng, lat)
source_mask_tensor = torch.Tensor(source_mask.reshape(1, source_lng, source_lat)).to(device)
print(f'Source city valid region num = [{np.sum(source_mask)}].')

# load target city data
target_data = np.load(f'./data/{args.target_city}/{args.travel_mode}{args.target_city}_{args.data_type}.npy')
target_lng, target_lat = target_data.shape[1], target_data.shape[2]
target_mask = target_data.sum(0) > 0
target_mask_tensor = torch.Tensor(target_mask.reshape(1, target_lng, target_lat)).to(device)
print(f'Target city valid region num = [{np.sum(target_mask)}].')

# raw traffic flow data -> [0, 1] normalization
source_data, source_max, source_min = min_max_normalize(source_data)
target_data, target_max, target_min = min_max_normalize(target_data)


# use history 6 time steps to predict the next 1 future time step
lag = [-6, -5, -4, -3, -2, -1]
# shape = (data_num, history_step, lng, lat)
source_train_x, source_train_y, source_val_x, source_val_y, source_test_x, source_test_y = split_x_y(source_data, lag)
target_train_x, target_train_y, target_val_x, target_val_y, target_test_x, target_test_y = split_x_y(target_data, lag)
# source city use the whole data, target city use 30/7/3 days data for training
if args.data_amount != 0:
    target_train_x = target_train_x[-args.data_amount * 24:, :, :, :]
    target_train_y = target_train_y[-args.data_amount * 24:, :, :, :]


# build dataset
source_train_dataset = TensorDataset(torch.Tensor(source_train_x), torch.Tensor(source_train_y))
source_val_dataset = TensorDataset(torch.Tensor(source_val_x), torch.Tensor(source_val_y))
source_test_dataset = TensorDataset(torch.Tensor(source_test_x), torch.Tensor(source_test_y))
target_train_dataset = TensorDataset(torch.Tensor(target_train_x), torch.Tensor(target_train_y))
target_val_dataset = TensorDataset(torch.Tensor(target_val_x), torch.Tensor(target_val_y))
target_test_dataset = TensorDataset(torch.Tensor(target_test_x), torch.Tensor(target_test_y))

# build dataloader
source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
source_val_loader = DataLoader(source_val_dataset, batch_size=args.batch_size, drop_last=True)
source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, drop_last=True)
target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, drop_last=True)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, drop_last=True)

# iteration num
len_src_train = len(source_train_loader)
len_tgt_train = len(target_train_loader)

# load contrastive positive pair
positive_pair = np.load(f'./data/{args.source_city}_{args.target_city}_pair.npy')

# contrastive learning temperature parameter
temperature = torch.ones([]) * args.temperature


# filter source features according to positive sample pairs
def select_feature(feature, pair):
    pair = torch.LongTensor(pair).to(feature.device)
    tensor_transpose = torch.transpose(feature, 0, 1)

    return tensor_transpose[pair.squeeze()].transpose(1, 0)


def cross_entropy(preds, targets):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss_ = (-targets * log_softmax(preds)).sum(1)

    return loss_


def clip_loss(source_feature, target_feature, t_mask):
    s_feature_selected = select_feature(source_feature, positive_pair)
    s_feature_selected = F.normalize(s_feature_selected, p=2, dim=-1)
    target_feature = F.normalize(target_feature, p=2, dim=-1)

    logits = torch.matmul(s_feature_selected, target_feature.transpose(1, 2)) / (temperature.exp())
    src_similarity = torch.matmul(s_feature_selected, s_feature_selected.transpose(1, 2))
    tgt_similarity = torch.matmul(target_feature, target_feature.transpose(1, 2))
    targets = F.softmax((src_similarity + tgt_similarity) / 2 * temperature, dim=-1)
    loss_transfer_sum = 0
    for j in range(logits.shape[0]):
        logits_slice = logits[j, :, :]
        targets_slice = targets[j, :, :]
        src_loss = cross_entropy(logits_slice, targets_slice)
        tgt_loss = cross_entropy(logits_slice.T, targets_slice.T)
        loss_transfer = (src_loss + tgt_loss) / 2.0
        loss_transfer = loss_transfer[t_mask.view(-1).bool()]
        loss_transfer_sum += torch.mean(loss_transfer)

    return loss_transfer_sum / logits.shape[0]


def train_model(net_, s_loader_, t_loader_, optimizer_, s_mask=None, t_mask=None):
    """
    s_feature shape = (batch, region, 256)
    t_feature shape = (batch, region, 256)
    s_pre shape = (batch, 1, valid_region)
    s_x shape = (batch, 6, lng, lat)
    s_y shape = (batch, 1, valid_region)
    """
    epoch_loss = []

    s_iter = iter(s_loader_)
    t_iter = iter(t_loader_)
    for i in range(0, len_src_train):
        s_x, s_y = next(s_iter)
        t_x, t_y = next(t_iter)

        # reset the target dataloader when the target data is used up
        if (i + 1) % len_tgt_train == 0:
            t_iter = iter(t_loader_)

        s_x = s_x.to(device)
        s_y = s_y.to(device)
        t_x = t_x.to(device)
        t_y = t_y.to(device)

        optimizer_.zero_grad()
        s_feature, t_feature, s_pre, t_pre = net_(s_x, t_x, s_mask.bool(), t_mask.bool(), return_feature=True)

        # source prediction loss
        batch_size_num = s_pre.shape[0]
        s_y = s_y.view(batch_size_num, 1, -1)[:, :, s_mask.view(-1).bool()]
        loss_pre_source = (s_pre - s_y) ** 2
        loss_pre_source = loss_pre_source.mean(0).sum()

        # target prediction loss
        t_y = t_y.view(batch_size_num, 1, -1)[:, :, t_mask.view(-1).bool()]
        loss_pre_target = (t_pre - t_y) ** 2
        loss_pre_target = loss_pre_target.mean(0).sum()

        # transfer loss
        loss_transfer_sum = clip_loss(s_feature, t_feature, t_mask)

        # model total loss
        total_loss = ((loss_pre_source + loss_pre_target) * (1.0 - args.transfer_weight) +
                      loss_transfer_sum * args.transfer_weight)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=args.max_norm)
        optimizer_.step()

        epoch_loss.append(total_loss.item())

    return epoch_loss


def evaluate_model(net_, loader_, mask_=None):
    with torch.no_grad():
        rmse = 0
        mae = 0
        valid_num = 0

        for it_ in loader_:
            (x, y) = it_
            x = x.to(device)
            y = y.to(device)   # shape = (B, 1, region)
            _, _, pre_, _ = net_(x, x, mask_.bool(), mask_.bool(), return_feature=True)
            valid_num += x.shape[0] * mask_.sum().item()
            batch_size_num = y.shape[0]
            lag_ = y.shape[1]
            y = y.view(batch_size_num, lag_, -1)[:, :, mask_.view(-1).bool()]

            rmse += ((pre_ - y) ** 2).sum().item()
            mae += (pre_ - y).abs().sum().item()

    return np.sqrt(rmse / valid_num), mae / valid_num   # RMSE, MAE


net = Sert(1, 3, target_mask_tensor, sigmoid_out=True).to(device)
print('Build network.')
print('---------------------------------------------------------------')
print(net)
print('---------------------------------------------------------------')

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

best_val_rmse = 999
best_test_rmse = 999
best_test_mae = 999
best_epoch = 0


# train model
start_time = time.time()
for epoch in range(args.epoch):
    net.train()
    loss = train_model(net, source_train_loader, target_train_loader, optimizer, source_mask_tensor, target_mask_tensor)
    print(f'Time = {(time.time() - start_time) // 60} min, epoch = [{epoch}], loss = {np.mean(loss)}')

    net.eval()
    # rmse_s_val, mae_s_val = evaluate_model(net, source_val_loader, source_mask_tensor)
    # rmse_s_val = rmse_s_val * (source_max - source_min)
    # mae_s_val = mae_s_val * (source_max - source_min)
    # print(f'Source val rmse = {rmse_s_val}, mae = {mae_s_val}')

    rmse_t_val, mae_t_val = evaluate_model(net, target_val_loader, target_mask_tensor)
    rmse_t_val = rmse_t_val * (target_max - target_min)
    mae_t_val = mae_t_val * (target_max - target_min)
    print(f'Target val rmse = {rmse_t_val}, mae = {mae_t_val}')

    rmse_t_test, mae_t_test = evaluate_model(net, target_test_loader, target_mask_tensor)
    rmse_t_test = rmse_t_test * (target_max - target_min)
    mae_t_test = mae_t_test * (target_max - target_min)
    print(f'Target test rmse = {rmse_t_test}, mae = {mae_t_test}')
    print('---------------------------------------------------------------')

    if rmse_t_val < best_val_rmse:
        best_val_rmse = rmse_t_val
        best_test_rmse = rmse_t_test
        best_test_mae = mae_t_test
        best_epoch = epoch

print(f'Best epoch = {best_epoch}.')
print(f'Best target val rmse = {best_val_rmse}')
print(f'Best target test rmse = {best_test_rmse}, mae = {best_test_mae}')

print(f'End SERT, from [{args.source_city} to {args.target_city}], use [{args.travel_mode} {args.data_type}] data,'
      f' with [{args.data_amount} days] of target data.')
