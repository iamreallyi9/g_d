import torch.nn as nn

import torch
from torch.utils.data import DataLoader
from gj_hourglass import HourglassModel
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from loaders.video_dataset import VideoDataset, VideoFrameDataset

from torch.autograd import Variable
from torchsummaryX import summary

import ts_loss
import small_model
from PIL import Image
import nyu_set
import numpy as np
import os
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

def load_s_net(file =False):

    new_model = small_model.gNet()
    if file == False:
        return new_model
    else:
        path = "./gj_dir/stu.pth"
        model_parameters = torch.load(path)
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

def label2target(label):
    targets = {}
    mask = torch.ones(label.shape)
    targets['gt_mask'] = mask.cuda()
    targets['depth_gt'] = label.cuda()
    return targets

def compare(batch=16,lr=0.01,epo=100):
    global T_mid_feature
    # 数据集
    #teacher——net
    net_t = load_t_net()
    net_t = nn.DataParallel(net_t)
    net_t = net_t.cuda()

    #student——net
    net_s = load_s_net(file =False)
    net_s = nn.DataParallel(net_s)
    net_s = net_s.cuda()

    train_Data = nyu_set.use_nyu_data(batch_s=batch, max_len=160, isBenchmark=False)
    writer1 = SummaryWriter('./gj_dir/train_pru_mod')
    optimizer = optim.Adam(net_s.parameters(), lr=0.001)

    Joint = ts_loss.JointLoss(opt=None).double().cuda()
    s_loss = ts_loss.SSIM().cuda()
    optimizer = optim.Adam(net_s.parameters(), lr=lr)


    net_t.eval()
    net_s.train()
    import time
    for epoch in range(epo):
        time_start = time.time()

        alpha = 0.7

        for i, data in enumerate( train_Data):
            images, labels = data
            images = Variable(images).double().cuda()
            target = label2target(labels)

            optimizer.zero_grad()

            #注册一个hook
            hh = net_t.module.seq[3].list[0][3].list[0][3].list[1][3].list[0][1].register_forward_hook(hook)

            output_t = net_t(images)[0].double()
            output_t = torch.div(1.0, torch.exp(output_t))
            output_s_depth,output_s_features= net_s(images)

            #注销hook
            hh.remove()
            TS_loss = 1-s_loss.forward(output_s_features,T_mid_feature[0])
            loss, loss1, loss2, loss3 = Joint(images, torch.log(output_s_depth), target)
            loss_all = loss+TS_loss
            loss_all.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.4f  A:%.4f  B:%.4f C:%.4f D:%.4f' % (
                epoch + 1, (i + 1) * batch, loss.item(), loss1, loss2, loss3, torch.min(output_s_depth).item()))

        writer1.add_scalar('loss', loss.item(), global_step=(epoch + 1))
        writer1.add_scalar('loss1', loss1, global_step=(epoch + 1))
        writer1.add_scalar('loss2', loss2, global_step=(epoch + 1))
        writer1.add_scalar('loss3', loss3, global_step=(epoch + 1))
        # debug_img = transforms.ToPILImage()(output_net)
        writer1.add_images('pre', output_s_depth, global_step=epoch)

        writer1.add_images('labels', labels, global_step=epoch)

        torch.save(net_s.state_dict(), 'gj_TS/stu.pth')
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')



if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    make_my_model()
    compare(batch=16,lr=0.01,epo=1)



