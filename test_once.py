import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from gj_hourglass import HourglassModel
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from loaders.video_dataset import VideoDataset, VideoFrameDataset
import torchvision.transforms as transforms

from utils.torch_helpers import to_device
from torchsummaryX import summary
import torch.autograd as autograd
import numpy as np
import ts_loss
import small_model
from PIL import Image
import gj_dataset
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

def load_s_net():
    new_model = small_model.gNet()
    new_model = torch.nn.DataParallel(new_model)
    #new_model = torch.nn.DataParallel(new_model).module
    model_file = "gj_TS/student_big_data.pth"
    #model_file = "gj_TS/0020.pth"
    #model_file = "gj_TS/student.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model

def id2image(id,trans):
    id = id.numpy()
    labels = []
    for i in id:
        path = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/depth/frame_{:06d}.png".format(i)
        image = np.array(Image.open(path)) / 255
        #image = trans(image)
        labels.append(image)
    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    print(labels.shape)
    #id =id.item()
    return labels

def see_t_net():
    # 记得修改batchsize
    #net = load_t_net()
    net =load_s_net()
    data_loader = gj_dataset.use_this_data()
    #data_loader = load_data()
    net.eval()
    num = 0
    for data in data_loader:
        num +=1
        data = to_device(data)
        stacked_images = data
        #stacked_images, metadata = data
        #frame_id = metadata["frame_id"][0]
        images = autograd.Variable(stacked_images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape

        C, H, W = shape[-3:]
        #images = images.reshape(-1, C, H, W).cuda().double()
        #images = images.reshape(-1,W,C,H).cuda().double()
        images = images.transpose(1,3)
        images = images.transpose(2,3)
        print(images.shape)
        # depth = nmodel.forward(stacked_images, metadata)
        # print(depth)
        prediction_d = net.forward(images)[0]  # 0is depth .1 is confidence

    
        print("================")
        out_shape = shape[:-3] + prediction_d.shape[-2:]
        print(out_shape)
        prediction_d = prediction_d.reshape(out_shape)

        prediction_d = torch.exp(prediction_d)
        depth = prediction_d.squeeze(-3)
        print(depth.shape)
        print("///////////////////")
        depth = depth.detach().cpu().numpy().squeeze()
        inv_depth = 1.0 / depth * 255
        im = Image.fromarray(inv_depth)
        if im.mode == "F":
            im = im.convert("L")
        im.save("gj_TS/"+str(num+1000)+".jpg")
        print("ok")


def hook(module, inputdata, output):
    global T_mid_feature
    T_mid_feature = []
    T_mid_feature.append(output.data)
    #print(output.data.shape)


def test_model():
    net = load_t_net()
    print(net)
    #net = HourglassModel(3)
    x = torch.randn(1,3,384,224).cuda()
    for param in net.named_parameters():
        print(param[0])

    hh0 = net.module.seq[3].list[0][3].list[0][3].list[1][2].register_forward_hook(hook)

    y =net(x)
    print("=++++++++++++++++++++=")
    summary(net.module,x)
    hh0.remove()


def make_my_model():
    net = small_model.gNet()
    x = torch.randn(1, 3, 384, 224)
    summary(net,x)

def compare():
    global T_mid_feature
    # 数据集
    data_loader =load_data()
    #teacher——net
    net_t = load_t_net()

    #student——net
    #net_s = load_s_net()
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
        batch_size = 2

        alpha = 0.7

        for i, data in enumerate( data_loader):
            images, labels = data
            #labels = id2image(labels['frame_id'],transf)

            #images = autograd.Variable(inputs.cuda(), requires_grad=False)
            # Reshape ...CHW -> XCHW
            shape = images.shape

            C, H, W = shape[-3:]
            images = images.reshape(-1, C, H, W)
            print(images.shape)
            images = images.to(device).double()
            #labels = labels.to(device).double()

            optimizer.zero_grad()

            #注册一个hook
            hh = net_t.module.seq[3].list[0][3].list[0][3].list[1][3].list[0][1].register_forward_hook(hook)

            output_t = net_t(images)[0].to(device)
            output_s_depth,output_s_features= net_s(images)
            output_s_features = output_s_features.to(device)
            output_s_depth = output_s_depth.to(device)

            #注销hook
            hh.remove()
            #loss1 = 1 - s_loss.forward(output_s_features, T_mid_feature[0])
            #loss2 = criterion(output_s_depth,output_t)
            loss1 = criterion(output_s_features, T_mid_feature[0])
            loss2 = 1 - s_loss.forward(output_s_depth,output_t)

            loss = loss1 * (1 - alpha) + loss2 * alpha
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.4f loss1: %.4f loss2: %.4f' % (
            epoch + 1, (i + 1) * batch_size, loss.item(), loss1.item(), loss2.item()))

        torch.save(net_s.state_dict(), 'gj_TS/student.pth')
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')
def see_raw():
    a= load_data()
    for i, data in enumerate( a):
        images,_=data
        debug_img = transforms.ToPILImage()(images[0, :, :, :].float().cpu())
        debug_img.save("see_raw.jpg")
        break


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    #make_my_model()
    #compare()
    #see_t_net()
    see_raw()

