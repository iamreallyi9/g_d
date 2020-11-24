from monodepth.mannequin_challenge.models import hourglass
from monodepth import mannequin_challenge_model as mcm
from torch.utils.data import DataLoader
from loaders.video_dataset import VideoDataset, VideoFrameDataset
import torch
from utils.torch_helpers import to_device
import os
from monodepth.depth_model_registry import get_depth_model
from torchsummaryX import summary
import torch.autograd as autograd
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

def pru_test():


    new_model = hourglass.HourglassModel(3)
    model_file = "checkpoints/test_mc.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)

    #new_model = torch.nn.DataParallel(new_model)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    new_model.eval()

    iut = torch.randn(1, 3, 384, 224)
    summary(new_model, iut)
    #self.seq//uncertainty_layer//pred_layer
    module = new_model.pred_layer
    #print(list(module.named_parameters()))
    #print(list(module.named_buffers()))

    #prune.random_unstructured(module，name = "weight", amount = 0.3)
    #prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    parameters_to_prune = (
        (new_model.pred_layer, 'weight'),
        (new_model.uncertainty_layer, 'weight'),
        (new_model.seq, 'weight')
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    prune.remove(new_model, 'weight')
    print("==========")
    iut = torch.randn(1, 3, 384, 224)
    summary(new_model, iut)



if __name__ == '__main__':
    pru_test()
