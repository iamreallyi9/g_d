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
import gj_dataset
import nyu_set
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from monodepth.mannequin_challenge.models.networks import LaplacianLayer,JointLoss
from torch.autograd import Variable
import ts_loss
from utils.torch_helpers import to_device

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
    model = model.to(device)
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
    torch.save(model.module, 'gj_dir/after.pth.tar')

def load_pru_mod(after_finetune =False):
    if after_finetune ==False:
        path = "./gj_dir/after.pth.tar"
    else:
        path = "./gj_dir/after_nyu.pth.tar"
    model = torch.load(path)
    return model

def benchmark_pruned():
    net = load_t_net(file=True)
    #net = load_pru_mod(after_finetune=True).double()
    net = nn.DataParallel(net)
    net = net.cuda()

    data_loader = nyu_set.use_nyu_data(batch_s=4,max_len=100,isBenchmark=True)
    writer1 = SummaryWriter('./gj_dir/benchmark_pru_mod')

    criterion = nn.MSELoss(reduction='mean').cuda()
    net.eval()

    num = 0
    for data,label in data_loader:
        num += 1
        target = label2target(label)
        images = Variable(data).double().cuda()

        prediction_d = net.forward(images)[0]  # 0is depth .1 is confidence

        e_rmse = JointLoss.compute_rmse_error(prediction_d, target)
        e_rel = JointLoss.compute_l1_rel_error(prediction_d,target)
        loss = criterion(prediction_d,target["depth_gt"])
        writer1.add_images('pre', prediction_d, global_step=num)

        writer1.add_scalar('rmse', e_rmse, global_step=num)
        writer1.add_scalar("rel",e_rel,global_step = num)
        writer1.add_scalar('loss', loss, global_step=num)

        writer1.add_images('label', label, global_step=num)

        print("ok")

def train_pru_mod(epoch =100,batch =4,lr=0.001):

    #net = load_t_net().double()
    net = load_pru_mod(after_finetune=True).double()
    net = nn.DataParallel(net)
    net = net.cuda()

    train_Data = nyu_set.use_nyu_data(batch_s=batch,max_len=160,isBenchmark=False)
    writer1 = SummaryWriter('./gj_dir/train_pru_mod')

    criterion = nn.MSELoss(reduction='mean').cuda()
    Joint = JointLoss(opt = None).double().cuda()
    s_loss = ts_loss.SSIM().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.train()
    import time
    for epoch in range(epoch):
        time_start = time.time()
        batch_size =batch

        for i, data in enumerate(train_Data):
            images ,depths = data
            # images = autograd.Variable(inputs.cuda(), requires_grad=False)
            images = Variable(images).double().cuda()
            depths = Variable(depths).double().cuda()

            # labels = labels.to(device).double()

            optimizer.zero_grad()
            # debug_img = transforms.ToPILImage()(images[0,:,:,:].float().cpu())
            # debug_img.save("debug.jpg")

            output_net = net(images)[0].double()

            # loss1 = 1 - s_loss.forward(output_s_features, T_mid_feature[0])
            # loss2 = criterion(output_s_depth,output_t)
            loss1 = criterion(output_net, depths)
            loss2 = Joint.LaplacianSmoothnessLoss(output_net,images)
            loss3 = Joint.compute_image_aware_2nd_smoothness_cost(output_net,images)
            #loss4 = Joint.compute_image_aware_1st_smoothness_cost(output_net,images)
            loss4 =1- s_loss.forward(output_net,depths)
            loss = loss1*10+loss2+loss3+loss4

            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.4f  A:%.4f  B:%.4f C:%.4f D:%.4f'  % (
                epoch + 1, (i + 1) * batch_size, loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()))

            writer1.add_scalar('loss', loss.item(), global_step=(epoch+1)*batch_size+i)
            writer1.add_scalar('loss2', loss2.item(), global_step=(epoch+1)*batch_size+i)
        #debug_img = transforms.ToPILImage()(output_net)
        writer1.add_images('pre',output_net,  global_step=epoch)
        shape = images.shape
        
        dep = torch.exp(output_net)
        
               
        dep = dep.detach().cpu().numpy()
        inv_dep = 1.0/dep*255
        
        writer1.add_images('pro-dep',inv_dep,  global_step=epoch)
        
        writer1.add_images('labels',depths,global_step=epoch)

        torch.save(net.module, "./gj_dir/after_nyu.pth.tar")
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')

def see_pru_mod():
    net = load_pru_mod(after_finetune=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x =torch.randn(1,3,384,224).to(device)
    summary(net,x)

def label2target(label):
    targets = {}
    mask = torch.ones(label.shape)
    targets['gt_mask'] = Variable(mask).double().cuda()
    targets['depth_gt'] = Variable(label).double().cuda()
    return targets


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    #train_pru_mod(epoch=301,batch=16,lr=0.00001)
    #load_pru_mod()
    #see_t_net()
    benchmark_pruned()
 
