import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUnit(nn.Module):   # Residual blocks without batch-norm
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = F.relu(x)
        z = self.conv1(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class STNet(nn.Module):   # the same spatial-temporal network as in CrossTReS
    def __init__(self, num_channels, num_convs, spatial_mask, sigmoid_out=False):
        super(STNet, self).__init__()
        self.num_channels = num_channels
        self.spatial_mask = spatial_mask.bool()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1)
        self.layers = []
        self.bns = []
        self.num_convs = num_convs
        for i in range(num_convs):
            self.layers.append(ResUnit(64, 64))
        self.layers = nn.ModuleList(self.layers)
        self.lstm = nn.LSTM(64, 128)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid_out = sigmoid_out

    def forward(self, x, spatial_mask=None, return_feat=False):
        if spatial_mask is None:
            spatial_mask = self.spatial_mask

        num_lag = (x.shape[1] // self.num_channels)
        batch_size = x.shape[0]
        outs = []

        for i in range(num_lag):
            input_data = x[:, i * self.num_channels:(i + 1) * self.num_channels, :, :]
            z = self.conv1(input_data)
            for layer in self.layers:
                z = layer(z)
            z = z.permute(0, 2, 3, 1).reshape(batch_size, -1, 64).contiguous()
            outs.append(z.view(-1, 64))

        z = torch.stack(outs, dim=0)
        temporal_out, (temporal_hid, _) = self.lstm(z)
        temporal_out = temporal_out[-1:, :]
        temporal = torch.cat([temporal_out.view(batch_size, -1, 128), temporal_hid.view(batch_size, -1, 128)], dim=-1)
        temporal_valid = temporal[:, spatial_mask.view(-1), :]   # removed regions with no traffic data
        hid = F.relu(self.linear1(temporal_valid))
        output = self.linear2(hid).permute(0, 2, 1)
        if self.sigmoid_out:
            output = torch.sigmoid(output)
        if return_feat:
            return temporal, output
        else:
            return output


class Siamese(nn.Module):   # a siamese network with two STNet
    def __init__(self, num_channels, num_convs, spatial_mask, sigmoid_out=False):
        super(Siamese, self).__init__()
        self.base_network = STNet(num_channels, num_convs, spatial_mask, sigmoid_out)

    def forward(self, source_input, target_input, source_mask, target_mask, return_feature=True):
        source_feature, source_pred = self.base_network(source_input, source_mask, return_feature)
        target_feature, target_pred = self.base_network(target_input, target_mask, return_feature)
        return source_feature, target_feature, source_pred, target_pred


class ProjectionHead(nn.Module):   # the projection head in contrastive learning
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False)
        )

    def forward(self, source_feature, target_feature):
        source_projection = self.projection_head(source_feature)
        target_projection = self.projection_head(target_feature)
        return source_projection, target_projection


class Sert(nn.Module):   # the main model
    def __init__(self, num_channels, num_convs, spatial_mask, sigmoid_out=False):
        super(Sert, self).__init__()
        self.siamese_net = Siamese(num_channels, num_convs, spatial_mask, sigmoid_out)
        self.projection_head = ProjectionHead()

    def forward(self, source_input, target_input, source_mask, target_mask, return_feature=True):
        """
        source_feature shape = (B, region, 256)
        target_feature shape = (B, region, 256)
        source_pred shape = (B, 1, valid_region)
        target_pred shape = (B, 1, valid_region)
        source_projection shape = (B, region, output_dim)
        target_projection shape = (B, region, output_dim)
        """
        source_feature, target_feature, source_pred, target_pred =\
            self.siamese_net(source_input, target_input, source_mask, target_mask, return_feature)

        source_projection, target_projection = self.projection_head(source_feature, target_feature)

        return source_projection, target_projection, source_pred, target_pred
