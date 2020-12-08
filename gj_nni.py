from nni.compression.torch import L1FilterPruner
from nni.compression.torch import TaylorFOWeightFilterPruner
from nni.compression.torch import ActivationMeanRankFilterPruner
from gj_hourglass import HourglassModel
import torch
from torchsummaryX import summary

def test_nni(model):

    config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
    }]
    pruner = ActivationMeanRankFilterPruner(model, config_list, statistics_batch_num=1)
    model = pruner.compress()
    print(model)

    x= torch.randn(1,3,384,224).cuda()
    summary(model,x)


def load_t_net():
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model

if __name__ == '__main__':
    model = load_t_net()
    test_nni(model)