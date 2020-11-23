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


def get_dep():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    depth_dir = 'esults/ayush/depth_mc/depth'

    # model = get_depth_model("mc")

    nmodel = mcm.MannequinChallengeModel()
    print(nmodel)
    # new_model = hourglass.HourglassModel(3)

    frames = [i for i in range(92)]

    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    nmodel.eval()

    # os.makedirs(depth_dir, exist_ok=True)
    for data in data_loader:
        data = to_device(data)
        stacked_images, metadata = data
        frame_id = metadata["frame_id"][0]

        # depth = nmodel.forward(stacked_images, metadata)
        # print(depth)
        depth = nmodel.estimate_depth(stacked_images)

        depth = depth.detach().cpu().numpy().squeeze()
        inv_depth = 1.0 / depth
    print(inv_depth)
    # print ("83")
    iut = torch.randn(1, 3, 384, 224)
    # summary(nmodel.model.netG, iut)


def only_g():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    depth_dir = 'esults/ayush/depth_mc/depth'
    frames = [i for i in range(92)]

    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    new_model = hourglass.HourglassModel(3)
    model_file = "checkpoints/mc.pth"
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
    print(inv_depth)

    m_path = 'pkl_model/my_pkl'
    torch.save(new_model,m_path)

def load():
    m_path = 'pkl_model/my_pkl'
    a=torch.load(m_path)
    print(a)

if __name__ == '__main__':
    #get_dep()
    only_g()
    load()