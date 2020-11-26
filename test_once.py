
import torch
from torch.utils.data import DataLoader
from monodepth.mannequin_challenge.models import hourglass
from torch.utils.tensorboard import SummaryWriter

import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset
from monodepth import mannequin_challenge_model as mcm
from utils.torch_helpers import to_device
from torchsummaryX import summary
import torch.autograd as autograd
import numpy as np

def test_model():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    frames = [i for i in range(92)]

    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )
    new_model = mcm.MannequinChallengeModel()
    #new_model = hourglass.HourglassModel(3)

    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    new_model = torch.nn.DataParallel(new_model)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    new_model.eval()

    # os.makedirs(depth_dir, exist_ok=True)
    for data in data_loader:
        data = to_device(data)
        stacked_images, metadata = data
        frame_id = metadata["frame_id"][0]
        images = autograd.Variable(stacked_images.cuda(), requires_grad=False)
        # Reshape ...CHW -> XCHW
        shape = images.shape

        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)
        # depth = nmodel.forward(stacked_images, metadata)
        # print(depth)
        prediction_d = new_model.forward(images)[0]  # 0is depth .1 is confidence

        out_shape = shape[:-3] + prediction_d.shape[-2:]
        prediction_d = prediction_d.reshape(out_shape)

        prediction_d = torch.exp(prediction_d)
        depth = prediction_d.squeeze(-3)

        depth = depth.detach().cpu().numpy().squeeze()
        inv_depth = 1.0 / depth
    print(inv_depth,np.shape(inv_depth))

if __name__ == '__main__':
    test_model()