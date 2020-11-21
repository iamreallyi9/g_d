from monodepth.mannequin_challenge.models import hourglass
from torch.utils.data import DataLoader
from loaders.video_dataset import VideoDataset, VideoFrameDataset
import torch
from utils.torch_helpers import to_device
import os
def get_dep():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    depth_dir = 'esults/ayush/depth_mc/depth'

    new_model = hourglass.HourglassModel(3)

    frames =[ i for i in range(92)]
    print(color_fmt,frames)
    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    #os.makedirs(depth_dir, exist_ok=True)
    for data in data_loader:
        data = to_device(data)
        stacked_images, metadata = data
        frame_id = metadata["frame_id"][0]
        depth = new_model.forward(stacked_images, metadata)

        depth = depth.detach().cpu().numpy().squeeze()
        inv_depth = 1.0 / depth
        print(inv_depth)

if __name__ == '__main__':
    get_dep()