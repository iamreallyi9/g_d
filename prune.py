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
import nni
from nni.compression.torch import LevelPruner, SlimPruner,FPGMPruner,AMCPruner
from nni.compression.torch.utils.counter import count_flops_params

def pru_test():


    new_model = hourglass.HourglassModel(3)
    model_file = "checkpoints/mc.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)

    new_model = torch.nn.DataParallel(new_model)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    new_model.eval()
    #self.seq//uncertainty_layer//pred_layer
    module = new_model.pred_layer
    print(list(module.named_parameters()))
    print(list(module.named_buffers()))
    
if __name__ == '__main__':
    pru_test()