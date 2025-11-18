import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.size()[0:2]
        y = self.avg_pool(x).view(b, c)
        mask = self.fc(y).view(b, c, 1)
        masked_output = x * mask.expand_as(x)
        return masked_output, mask


class FrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, stride=2)
        self.conv1 = nn.Conv2d(8, 16, 3, stride=2)
        # self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        frame_num = x.shape[1]

        features = []
        for i in range(frame_num):
            feat = self.conv0(x[:, i])
            feat = self.maxpool(feat)
            feat = self.relu(feat)

            feat = self.conv1(feat)
            feat = self.maxpool(feat)
            feat = self.relu(feat)

            # feat = self.conv2(feat)
            # feat = self.maxpool(feat)
            # feat = self.relu(feat)

            feat = self.avgpooling(feat)
            features.append(feat)
        features = torch.stack(features, dim=1)

        return features


class Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.enc1 = FrameEncoder()
        self.enc2 = FrameEncoder()
        self.enc3 = FrameEncoder()

        self.se = SELayer(10)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(480, 120)
        self.linear2 = nn.Linear(120, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, frames):
        roi1 = frames.get("roi1")  # [B, N, C, H, W] (N: frame num)
        roi2 = frames.get("roi2")
        roi3 = frames.get("roi3")

        feat1 = self.enc1(roi1).squeeze()
        feat2 = self.enc1(roi2).squeeze()
        feat3 = self.enc1(roi3).squeeze()

        feat = torch.concat([feat1, feat2, feat3], dim=2)

        masked_output, mask = self.se(feat)

        feat = self.flatten(masked_output)
        feat = self.relu(self.linear1(feat))
        out = self.relu(self.linear2(feat))

        return out
