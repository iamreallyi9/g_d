# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.nn.functional as F
import torch
import torch.nn as nn


class inception(nn.Module):
    def __init__(self, input_size, config):
        self.config = config
        super(inception, self).__init__()
        self.convs = nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(nn.Sequential(
            nn.Conv2d(input_size, config[0][0], 1),
            nn.BatchNorm2d(config[0][0], affine=False),
            nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filt = config[i][0]
            pad = int((filt-1)/2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = nn.Sequential(
                nn.Conv2d(input_size, out_a, 1),
                nn.BatchNorm2d(out_a, affine=False),
                nn.ReLU(True),
                nn.Conv2d(out_a, out_b, filt, padding=pad),
                nn.BatchNorm2d(out_b, affine=False),
                nn.ReLU(True)
            )
            self.convs.append(conv)

    def __repr__(self):
        return "inception"+str(self.config)

    def forward(self, x):
        ret = []
        for conv in (self.convs):
            ret.append(conv(x))
        return torch.cat(ret, dim=1)


class Channels1(nn.Module):
    def __init__(self):
        super(Channels1, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
            )
        )  # EE
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # EEE

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels2(nn.Module):
    def __init__(self):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])
            )
        )  # EF
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels1(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # EE1EF

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels3(nn.Module):
    def __init__(self):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                Channels2(),
                inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # BD2EG
        self.list.append(
            nn.Sequential(
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])
            )
        )  # BC

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)


class Channels4(nn.Module):
    def __init__(self):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(
            nn.Sequential(
                nn.AvgPool2d(2),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                Channels3(),
                inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
                inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]]),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )
        )  # BB3BA
        self.list.append(
            nn.Sequential(
                inception(128, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
            )
        )  # A

    def forward(self, x):
        return self.list[0](x)+self.list[1](x)

#这是原来的model，被替换了
class ori_HourglassModel(nn.Module):
    def __init__(self, num_input):
        super(HourglassModel, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(num_input, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Channels4(),
        )

        uncertainty_layer = [
            nn.Conv2d(64, 1, 3, padding=1), torch.nn.Sigmoid()]
        self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
        self.pred_layer = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, input_):
        pred_feature = self.seq(input_)

        pred_d = self.pred_layer(pred_feature)
        pred_confidence = self.uncertainty_layer(pred_feature)

        return pred_d, pred_confidence

class HourglassModel(nn.Module):
    def __init__(self, num_input):
        super(HourglassModel, self).__init__()
        print("num input is 3")

        self.cfg = [(1, 16, 1, 1),
                    (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                    (6, 32, 3, 2),
                    (6, 64, 4, 2),
                    (6, 96, 3, 1),
                    (6, 160, 3, 2),
                    (6, 320, 1, 1)]

        self.layer1 = Block(3, 6, 1, 1)
        self.layer2 = Block(6, 12, 2, 2)
        self.layer3 = Block(12, 24, 4, 2)
        self.layer4 = Block(24, 36, 3, 1)
        self.layer5 = Block(36, 64, 3, 2)
        self.layer6 = nn.Conv2d(64, 256, 1, 1)
        self.layer7 = nn.Conv2d(256, 64, 1, 1)
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        features = self.layer1(x)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.layer5(features)
        features = self.layer6(features)

        depth = self.layer7(features)
        depth = self.decoder(depth)
        return depth, features

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out