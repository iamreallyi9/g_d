from gj_hourglass import HourglassModel
from torchsummaryX import summary
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os.path
from utils.torch_helpers import to_device
import torch.autograd as autograd
from PIL import Image

def new_prune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            # 1 input image channel, 6 output channels, 3x3 square conv kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, int(x.nelement() / x.shape[0]))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = LeNet().to(device=device)
    x = torch.randn(1,1,28,28).to(device)
    summary(model,x)


    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )
    x = torch.randn(1, 1, 28, 28).to(device)
    summary(model, x)

def load_t_net(file = False):
    new_model = HourglassModel(3)
    new_model = torch.nn.DataParallel(new_model)
    if file==True:
        #model_file = "results/ayush/R_hierarchical2_mc/B0.1_R1.0_PL1-0_LR0.0004_BS4_Oadam/checkpoints/0002.pth"
        model_file = "0020.pth"
        #model_file = 'gj_dir/after.pth'
        model_parameters = torch.load(model_file)
        new_model.load_state_dict(model_parameters)
    return new_model

def test_big():
    model = load_t_net(file=True)
    #print(model.module.state_dict().key())
    #print(list(model.module.seq[0].named_parameters()))
    #print(list(model.module.named_modules()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,3,384,224).to(device)
    summary(model,x)
    num =0
 
    for name ,mod in model.module.named_modules():
        
        num+=1
        if num %40==0:
            continue
        if isinstance(mod,torch.nn.Conv2d):
            print("yes")
            prune.l1_unstructured(mod,name='weight',amount=0.5)
        else:
            print("no")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    parameters_to_prune = (
        (model.module.seq[3].list[0][3].list[0][3].list[0][1].convs[2][3], 'weight'),
        (model.module.seq[3].list[0][3].list[0][3].list[0][1].convs[3][0], 'weight'),
        (model.module.seq[3].list[0][3].list[1][1].convs[0][0],'weight'),
        (model.module.seq[0],'weight')    

)
    #prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
    x = torch.randn(1, 3, 384, 224).to(device)
    summary(model, x)
    #prune.remove(model,'weight')
    torch.save(model, 'gj_dir/after.pth.tar')

def load_pru_mod():
    path = "./gj_dir/after.pth.tar"
    model = torch.load(path)
    return model


def see_t_net():
    # 记得修改batchsize
    # net = load_t_net()
    net = load_pru_mod()
    data_loader = gj_dataset.use_this_data()
    # data_loader = load_data()
    net.eval()
    num = 0
    for data in data_loader:
        num += 1
        data = to_device(data)
        stacked_images = data
        # stacked_images, metadata = data
        # frame_id = metadata["frame_id"][0]
        images = autograd.Variable(stacked_images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape

        C, H, W = shape[-3:]
        # images = images.reshape(-1, C, H, W).cuda().double()
        # images = images.reshape(-1,W,C,H).cuda().double()
        images = images.transpose(1, 3)
        images = images.transpose(2, 3)
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
        im.save("gj_TS/" + str(num + 1000) + ".jpg")
        print("ok")


if __name__ == '__main__':
    load_pru_mod()
