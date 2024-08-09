"""
Input: images, trained SimCLR model (poi / geo)
Purpose: extract image features (poi / geo)
Output: image features (poi / geo)
"""

import os
import argparse
import torch
import torch.utils.data
import time
from dataloader import *
from utils import yaml_config_hook

from simclr import SimCLR
from simclr.modules import get_resnet
from simclr.modules.transformations import TransformsSimCLR


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

view = 'geo'   # poi / geo   TODO: change view
city = 'nyc'   # nyc / chi / dc   TODO: change city
trained_model_path = f'./result/model_checkpoint/{view}_view/'
# save_feature_path = f'./result/feature/{view}_view_feature.npy'   # bj
save_feature_path = f'./result/feature/3_city/{city}/{city}_{view}.npy'


def inference(loader, simclr_model, device):
    feature_vector = []

    for step, (x, y) in enumerate(loader):
        x = x.to(device)   # anchor image

        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)

        h = h.detach()   # encoder output as image feature
        feature_vector.extend(h.cpu().detach().numpy())

        # if step % 20 == 0:
        #     print(f"Step = [{step}/{len(loader)}], extracting features...")

    feature_vector = np.array(feature_vector)
    print("*** Features shape = {} ***".format(feature_vector.shape))
    return feature_vector


if __name__ == "__main__":
    print(f'GPU is available = {torch.cuda.is_available()}')
    print(f'GPU device = cuda: {torch.cuda.current_device()}')

    parser = argparse.ArgumentParser(description='SimCLR')
    config = yaml_config_hook('config.yaml')
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MyDataset(
        # './dataset/image_id.npy',
        # './dataset/pos_pair_poi.npy',   # no use here
        # './dataset/images/bj/',
        f'./dataset/images/{city}_id.npy',
        f'./dataset/images/{city}_id.npy',
        f'./dataset/images/{city}/',
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features

    # load pre-trained model from checkpoint
    print('*** Load pre-trained model ***')
    model = SimCLR(encoder, args.projection_dim, n_features)

    model_fp = os.path.join(trained_model_path, 'checkpoint_150.tar')
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()

    # extract feature
    print(f'*** Extract {city} features in {view} view ***')
    start_time = time.time()

    image_feature = inference(train_loader, model, args.device)
    np.save(save_feature_path, image_feature)   # save the image embeddings

    end_time = time.time()
    print(f'Time = {end_time - start_time} s.')

