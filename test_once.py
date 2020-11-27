import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from gj_hourglass import HourglassModel
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import optimizer
from loaders.video_dataset import VideoDataset, VideoFrameDataset
import torchvision.transforms as transforms
from monodepth import mannequin_challenge_model as mcm
from utils.torch_helpers import to_device
from torchsummaryX import summary
import torch.autograd as autograd
import numpy as np
import ts_loss
import small_model
from PIL import Image
import numpy as np
T_mid_feature=[]

def load_data():
    color_fmt = 'results/ayush/color_down/frame_{:06d}.raw'
    frames = [i for i in range(92)]
    dataset = VideoFrameDataset(color_fmt, frames)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )
    return data_loader

def load_t_net():
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model

def id2image(id):
    id =id.item()
    path = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth/frame_{:06d}.png".format(id)
    image= np.array(Image.open(path))/255
    return image

def test():

    net = small_model.AutoEncoder()
    tnet = load_t_net()
    x = torch.randn(1,3,384,224)
    y = tnet(x)
    print(y.size(),y.type())
    print("djdjdjdhdhdhdfhddhdhdhdhdh")
    print(y)
    #summary(tnet,x)

def hook(module, inputdata, output):
    T_mid_feature = []
    T_mid_feature.append(output.data)


def test_model():
    net = load_t_net()
    #net = HourglassModel(3)
    x = torch.randn(1,3,384,224)
    for param in net.named_parameters():
        #print(param[0])
        pass
    hh = net.module.seq[3].list[0][3].list[0][3].list[1][3].list[0][1].register_forward_hook(hook)
    y =net(x)
    hh.remove()
    print(T_mid_feature)

def make_my_model():
    net = small_model.gNet()
    x = torch.randn(1, 3, 384, 224)
    summary(net,x)

def compare():
    # 数据集
    data_loader =load_data()
    #teacher——net
    net_t = load_t_net()

    #student——net
    net_s = small_model.gNet()
    net_s = nn.DataParallel(net_s)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss(reduction='mean').to(device)
    criterion2 = nn.KLDivLoss()

    optimizer = optim.Adam(net_s.parameters(), lr=0.001)


    print(device)
    net_s.to(device)
    net_t.to(device)

    s_loss = ts_loss.SSIM().to(device)
    transf=transforms.ToTensor()
    net_t.eval()
    net_s.train()
    import time
    for epoch in range(1):
        time_start = time.time()
        running_loss = 0.
        batch_size = 1

        alpha = 0.5

        for i, data in enumerate( data_loader):
            images, labels = data
            labels = id2image(labels['frame_id'])
            labels=transf(labels)

            #images = autograd.Variable(inputs.cuda(), requires_grad=False)
            # Reshape ...CHW -> XCHW
            shape = images.shape

            C, H, W = shape[-3:]
            images = images.reshape(-1, C, H, W)

            images = images.to(device).double()
            labels = labels.to(device).double()

            optimizer.zero_grad()

            #注册一个hook
            hh = net_t.module.seq[3].list[0][3].list[0][3].list[1][3].list[0][1].register_forward_hook(hook)

            output_t = net_t(images)[0].to(device)
            output_s_depth,output_s_features= net_s(images).to(device)

            #注销hook
            hh.remove()

            loss1 = criterion(output_s_features, T_mid_feature[0])
            loss2 = 1 - s_loss.forward(output_s_depth,output_t)

            loss = loss1 * (1 - alpha) + loss2 * alpha
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' % (
            epoch + 1, (i + 1) * batch_size, loss.item(), loss1.item(), loss2.item()))

        torch.save(net_s, 'gj_TS/student.pkl')
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    #make_my_model()
    compare()
    #test_model()
    #test()
