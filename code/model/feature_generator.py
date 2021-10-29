import torch
import torch.nn as nn
import torch.nn.functional as F


class feature_generator(nn.Module):
    def __init__(self,configs):
        super(feature_generator, self).__init__()
        # self.configs = configs
        self.conv1 = nn.Conv3d(in_channels=2,
                               out_channels=64,
                               kernel_size=(11,11,3),
                               stride=2,
                               padding=2)
        self.conv2 = nn.Conv3d(in_channels=64,
                               out_channels=128,
                               kernel_size=(5,5,3),
                               stride=1,
                               padding=2)
        self.conv3 = nn.Conv3d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3,3,3),
                               stride=1,
                               padding=0)
        self.conv4 = nn.Conv3d(in_channels=64,
                               out_channels=configs[1],
                               kernel_size=(3,3,3),
                               stride=1,
                               padding=0)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(configs[1])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = F.relu(self.bn2(self.conv2(out)), inplace=False)
        out = F.relu(self.bn3(self.conv3(out)), inplace=False)
        out = F.relu(self.bn4(self.conv4(out)), inplace=False)
        return out


# embedding = feature_generator([32, 64, 9, 9, 120])

# inpu = torch.zeros((32, 2, 32, 32, 4))

# out = embedding(inpu)
# print(out.shape)


