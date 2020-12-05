from gj_hourglass import HourglassModel
from torchsummaryX import summary
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os.path
def new_prune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square conv kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, int(x.nelement() / x.shape[0]))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = LeNet().to(device=device)
    x = torch.randn(1,1,28,28).to(device)
    summary(model,x)


    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    x = torch.randn(1, 1, 28, 28).to(device)
    summary(model, x)

def load_t_net(file = False):
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    if file==True:
        model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
        model_parameters = torch.load(model_file)
        new_model.load_state_dict(model_parameters)
    return new_model

def test_big():
    model = load_t_net()
    print(list(model.module.seq[3].named_parameters()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===========================")
    x=torch.randn(1,3,384,224).to(device)
    summary(model,x)

    parameters_to_prune = (
        (model.module.seq[3].list[0][3].list[0][3].list[0][1].convs[2][3], 'weight'),
        (model.module.seq[3].list[0][3].list[0][3].list[0][1].convs[3][0], 'weight'),
        (model.module.seq[3].list[0][3].list[0][3].list[0][1].convs[1][0], 'weight'),
        (model.module.seq[0].conv,'weight')
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    x = torch.randn(1, 3, 384, 224).to(device)
    summary(model, x)

if __name__ == '__main__':
    test_big()
