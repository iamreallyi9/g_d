import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from monodepth.mannequin_challenge.models import hourglass
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset
from monodepth import mannequin_challenge_model as mcm
from utils.torch_helpers import to_device
from torchsummaryX import summary
import torch.autograd as autograd
import numpy as np
from small_model import CNN
import small_model
def load_data():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    frames = [i for i in range(92)]
    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )
    return data_loader

def load_t_net():
    new_model = hourglass.HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model

def test():

    net = small_model.MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    summary(net,x)
    print(y.size())

def test_model():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    frames = [i for i in range(92)]
    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )
    #new_model = mcm.MannequinChallengeModel()
    new_model = hourglass.HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)

    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)


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

def compare():
    # 数据集
    data_loader =load_data()
    #teacher——net
    net_t = load_t_net()
    #student——net
    net_s = CNN()

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()
    optimizer = optim.Adam(net_s.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_s.to(device)
    net_t.to(device)

    import time
    for epoch in range(100):
        time_start = time.time()
        running_loss = 0.
        batch_size = 4

        alpha = 0.95

        for i, data in enumerate( data_loader):
            inputs, labels = data
            images = autograd.Variable(inputs.cuda(), requires_grad=False)
            # Reshape ...CHW -> XCHW
            shape = images.shape

            C, H, W = shape[-3:]
            images = images.reshape(-1, C, H, W)

            soft_target = net_t(images)

            optimizer.zero_grad()

            outputs = net_s(images)

            loss1 = criterion(outputs, labels)

            T = 2
            outputs_S = F.log_softmax(outputs / T, dim=1)
            outputs_T = F.softmax(soft_target / T, dim=1)

            loss2 = criterion2(outputs_S, outputs_T) * T * T

            loss = loss1 * (1 - alpha) + loss2 * alpha

            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' % (
            epoch + 1, (i + 1) * batch_size, loss.item(), loss1.item(), loss2.item()))

        torch.save(net_s, '/gj_TS/student.pkl')
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')

if __name__ == '__main__':
    #test_model()
    test()