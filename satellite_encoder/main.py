"""
Input: images, positive samples
Purpose: train SimCLR by contrastive learning
Output: SimCLR model parameters
"""

import os
import argparse
import time
from dataloader import *
from utils import yaml_config_hook, load_optimizer, save_model

import torch
import torch.utils.data

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

view = 'poi'   # poi / geo   TODO: change view
positive_sample_path = f'./dataset/pos_pair_{view}.npy'
model_save_path = f'./result/model_checkpoint/{view}_view/'


def train(train_loader, model, criterion, optimizer):
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):   # step = 12200 (image num) / 128 (batch_size)
        optimizer.zero_grad()

        x_i = x_i.cuda()
        x_j = x_j.cuda()
        h_i, h_j, z_i, z_j = model(x_i, x_j)   # h: encoder output, z: projection head output

        loss = criterion(z_i, z_j)   # use z to compute loss
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        #     print(f'Step = [{step}/{len(train_loader)}], Loss = {loss.item()}')

        loss_epoch += loss.item()

    return loss_epoch


def main(arg):
    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)

    train_dataset = MyDataset(
        './dataset/image_id.npy',
        positive_sample_path,  # poi / geo
        './dataset/images/bj/',
        transform=TransformsSimCLR(size=arg.image_size),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=arg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        sampler=None,
    )
    print(f'*** Load dataset, image number = {len(train_dataset)} ***')

    encoder = get_resnet(arg.resnet, pretrained=False)
    n_features = encoder.fc.in_features   # SimCLR projection head input feature dim

    print('*** Build model ***')
    model = SimCLR(encoder, arg.projection_dim, n_features)
    model = model.to(arg.device)

    optimizer, scheduler = load_optimizer(arg, model)   # scheduler is automatic lr
    criterion = NT_Xent(arg.batch_size, arg.temperature, arg.world_size)   # InfoNCE loss

    print(model)
    print('----------------------------------------------------------')

    print('*** Start train ***')
    best_loss = float('inf')
    best_model = None

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(train_loader, model, criterion, optimizer)

        if scheduler:
            scheduler.step()

        # only save the min loss model
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            if best_model is not None:
                os.remove(best_model)
            best_model = save_model(model_save_path, model)
            print(f'Epoch = [{epoch}/{arg.epochs}], Loss = {loss_epoch / len(train_loader)}, lr = {round(lr, 5)}')

    print('*** Saved model ***')
    print('*** End train ***')


if __name__ == "__main__":
    print(f'GPU is available = {torch.cuda.is_available()}')
    print(f'GPU device = cuda: {torch.cuda.current_device()}')

    parser = argparse.ArgumentParser(description='SimCLR')
    config = yaml_config_hook('config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.world_size = 1

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f'Time = {end_time - start_time} s.')
