import sys
sys.path.append("..") # 这句是为了导入nyu_set
import torch
import nyu_set
import torch.nn as nn
import torch.optim as optim
import tensorboard as SummaryWriter
import ts_loss
from ts_loss import JointLoss
from torch.autograd import Variable
import prune
from p2net import networks
import os
#剪枝后的
path_p_e = "./nni/speed_mod.pth.tar"
#fine-tune后的
path_e = "./nni/nni_f_e.pth.tar"
path_d = "./nni/nni_f_d.pth.tar"
path_sum = './nni/writer_coder'
def load_encoder(after_f = False):
    if after_f ==False:
        net = torch.load(path_p_e)
    else:
        net = torch.load(path_e)
    return net

def load_decoder(after_f = False):
    if after_f ==False:
        decoder = networks.resnet_encoder.gDecoder()
    else:
        decoder = torch.load(path_d)

def train_pru_mod(epoch=100, batch=4, lr=0.001):
    # net = load_t_net().double()
    net_e = load_encoder()
    net_e = nn.DataParallel(net_e).cuda()
    net_d = load_decoder()
    net_d = nn.DataParallel(net_d).cuda()


    train_Data = nyu_set.use_nyu_data(batch_s=batch, max_len=160, isBenchmark=False)
    writer1 = SummaryWriter(path_sum)

    Joint = JointLoss(opt=None).double().cuda()
    s_loss = ts_loss.SSIM().cuda()

    opt_e = optim.Adam(net_e.parameters(), lr=lr)
    opt_d = optim.Adam(net_d.parameters(), lr=lr)

    net_e.train()
    net_d.train()
    import time
    for epoch in range(epoch):
        time_start = time.time()
        batch_size = batch

        for i, data in enumerate(train_Data):
            images, depths = data

            images = Variable(images).double().cuda()
            target = prune.label2target(depths)

            tmp = net_e(images)
            out = net_d(tmp)

            opt_e.zero_grad()
            opt_d.zero_grad()

            loss,loss1,loss2,loss3 = Joint(images,torch.log(out),target)

            loss.backward()
            opt_d.step()
            opt_e.step()

            print('[%d, %5d] loss: %.4f  A:%.4f  B:%.4f C:%.4f D:%.4f' % (
                epoch + 1, (i + 1) * batch_size, loss.item(), loss1, loss2, loss3, torch.min(out).item()))

        writer1.add_scalar('loss', loss.item(), global_step=(epoch + 1))
        writer1.add_scalar('loss1', loss1, global_step=(epoch + 1))
        writer1.add_scalar('loss2', loss2, global_step=(epoch + 1))
        writer1.add_scalar('loss3', loss3, global_step=(epoch + 1))
        # debug_img = transforms.ToPILImage()(output_net)
        writer1.add_images('pre', out, global_step=epoch)

        writer1.add_images('labels', depths, global_step=epoch)

        torch.save(net_e.module, path_e)
        torch.save(net_d.module, path_d)
        time_end = time.time()
        print('Time cost:', time_end - time_start, "s")

    print('Finished Training')

if __name__ == '__main__':
    #torch.set_default_tensor_type(torch.DoubleTensor)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    train_pru_mod()