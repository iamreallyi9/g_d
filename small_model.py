import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
from monodepth.mannequin_challenge.options.train_options import TrainOptions

class gNet(nn.Module):
    def __init__(self):
        super(gNet, self).__init__()
        self.cfg = [(1, 16, 1, 1),
               (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
               (6, 32, 3, 2),
               (6, 64, 4, 2),
               (6, 96, 3, 1),
               (6, 160, 3, 2),
               (6, 320, 1, 1)]

        self.layer1 = Block(3,6,1,1)
        self.layer2 = Block(6,12,2,2)
        self.layer3 = Block(12,24,4,2)
        self.layer4 = Block(24,36,3,1)
        self.layer5 = Block(36,64,3,2)
        self.layer6 = nn.Conv2d(64,256,1,1)
        self.layer7 = nn.Conv2d(256,64,1,1)
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
        depth =self.decoder(depth)
        return depth,features


class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1


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


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)


    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)

        return out


def SmallMCM():
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self):
        #super().__init__()
        parser = TrainOptions()
        parser.initialize()
        params = parser.parser.parse_args(["--input", "single_view"])
        params.isTrain = False

        model_file = "/data/consistent_depth/gj_TS/student.pth"
        self.model =gNet()
        model_parameters = torch.load(model_file)
        self.model.load_state_dict(model_parameters)
        print("load the pth ->ok")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        print("want to see the parameters")
        return self.model.parameters()

    def estimate_depth(self, images):
        images = autograd.Variable(images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape
        "here----"
        print(shape)
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)
        print("shape is kkkkkkkk")
        print(images.shape)
        depth, _ = self.model.forward(images)

        # Reshape X1HW -> BNHW
        out_shape = shape[:-3] + depth.shape[-2:]
        depth = depth.reshape(out_shape)

        depth = torch.exp(depth)
        depth = depth.squeeze(-3)

        return depth

    def save(self, file_name):
        print(file_name)
        file_name = "gj_TS/"+file_name
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_name)
