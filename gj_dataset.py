import torch.utils.data as data
import os
import io
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from gj_hourglass import HourglassModel
import small_model
import torch.nn as nn
import torch.optim as optim
import ts_loss
import torchvision.transforms as transforms

def load_t_net():
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0020.pth"
    model_parameters = torch.load(model_file)
    new_model.load_state_dict(model_parameters)
    return new_model

def hook(module, inputdata, output):
    global T_mid_feature
    T_mid_feature = []
    T_mid_feature.append(output.data)
    #print(output.data.shape)

def compare():
    global T_mid_feature
    # 数据集
    data_loader =use_this_data()
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

        alpha = 0.7

        for i, data in enumerate( data_loader):
            images = data
            #labels = id2image(labels['frame_id'],transf)

            #images = autograd.Variable(inputs.cuda(), requires_grad=False)
            # Reshape ...CHW -> XCHW
            shape = images.shape

            C, H, W = shape[-3:]
            images = images.reshape(-1, C, H, W)

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

        torch.save(net_s.module.state_dict(), 'gj_TS/student.pth')
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')

class G_DataSet(data.Dataset):
    def __init__(self,img_path,mask_img_path,transform=None):
        self.img_path = img_path
        self.mask_img_path = mask_img_path
        self.transform = transform
        self.mask_img = os.listdir(mask_img_path)

    def __len__(self):
        return len(self.mask_img)
    def __getitem__(self, idx):
        label_img_name = self.mask_img[idx]
        label_img_name = "nyu_images/"+label_img_name

        input_img = cv2.imread(label_img_name, cv2.IMREAD_COLOR)
        input_img=input_img/255

        if self.transform:
            input_img = self.transform(input_img)
        return input_img
def use_this_data():
    img_path = 'nyu_images'
    mask_img_path = 'nyu_images'
    d = G_DataSet(img_path, mask_img_path)
    data_loader = DataLoader(
        d, batch_size=1, shuffle=False, num_workers=4
    )
    return data_loader

if __name__ == '__main__':

    compare()


