
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("..") # 这句是为了导入nyu_set

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch.nn as nn
import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
import nyu_set
from tensorboardX import SummaryWriter
from monodepth.monodepth2.layers import compute_depth_errors
import torch.autograd as autograd
from monodepth.mannequin_challenge.models.networks import LaplacianLayer,JointLoss

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
    return decoder

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on one Single Image.')
    parser.add_argument('--image_path', type=str,
                        default='./asserts/sample.png',
                        help='path to a test image')
    parser.add_argument('--model_name', type=str,
                        default='weights_5f',
                        help='name of a pretrained model to use',
                        choices=[
                            "weights_3f",
                            "weights_5f", ])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def prepare_model_for_test(args, device):
    model_path = args.model_name
    print("-> Loading model from ", model_path)
    model_path = os.path.join("ckpts", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    encoder = networks.ResnetEncoder(18, False)
    decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=range(1),
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict(decoder_dict)

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()

    return encoder, decoder, encoder_dict['height'], encoder_dict['width']


def inference(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    encoder, decoder, thisH, thisW = prepare_model_for_test(args, device)
    
    data_loader = nyu_set.use_nyu_data(batch_s=1, max_len=400, isBenchmark=True)
    writer1 = SummaryWriter('/data/consistent_depth/gj_dir/benchmark_pp')

    with torch.no_grad():
        num = 0
        su = 0
        for data,label in data_loader:
            num +=1
            label = label.cpu()
            input_image = transforms.ToPILImage()(data[0])
            original_width, original_height = input_image.size
            input_image = input_image.resize((thisH, thisW), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            input_image = input_image.to(device)
            outputs = decoder(encoder(input_image))

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            #print(torch.max(disp_resized),torch.min(disp_resized))
            #disp_resized, _ = disp_to_depth(disp_resized, 0.1, 10)
            #print(torch.max(disp_resized),torch.min(disp_resized))
            #disp_resized = torch.div(1.0,disp_resized)
            disp_resized = torch.div(disp_resized,torch.max(disp_resized))
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(label[0], disp_resized.cpu())

            for ind, ten in enumerate(disp_resized):
                #disp_resized[ind] = torch.div(disp_resized[ind], torch.max(disp_resized[ind]))
                pass
                # prediction_d = torch.mul(255,prediction_d)

            writer1.add_images('pre', disp_resized, global_step=num)

            writer1.add_scalar('rmse', rmse, global_step=num)
            writer1.add_scalar("abs_rel", abs_rel, global_step=num)
            writer1.add_scalar('sq_rel', sq_rel, global_step=num)
            writer1.add_scalar('rmse_log', rmse_log, global_step=num)
            writer1.add_scalar('a1', a1, global_step=num)
            writer1.add_scalar('a2', a2, global_step=num)
            writer1.add_scalar('a3', a3, global_step=num)
           
            writer1.add_images('label', label, global_step=num)
            su += a3.item()
            print(su/num)
        #scaled_disp, _ = disp_to_depth(disp, 0.1, 10)
        # Saving colormapped depth image
        #vmax = np.percentile(disp_resized_np, 95)
    writer1.close()
    print('-> Done!')
def mark_pru():
    net_e = load_encoder(after_f=True)
    net_e = nn.DataParallel(net_e).cuda()
    net_d = load_decoder(after_f=True)
    net_d = nn.DataParallel(net_d).cuda()

    data_loader = nyu_set.use_nyu_data(batch_s=1, max_len=400, isBenchmark=True)
    writer1 = SummaryWriter('/data/consistent_depth/gj_dir/benchmark_p2')

    with torch.no_grad():
        num = 0
        su = 0
        for data,label in data_loader:
            num +=1
            data = autograd.Variable(data.double().cuda(), requires_grad=False)


            prediction_d = net_d(net_e(data))

            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(label, prediction_d)

            writer1.add_images('pre', prediction_d, global_step=num)

            writer1.add_scalar('rmse', rmse, global_step=num)
            writer1.add_scalar("abs_rel", abs_rel, global_step=num)
            writer1.add_scalar('sq_rel', sq_rel, global_step=num)
            writer1.add_scalar('rmse_log', rmse_log, global_step=num)
            writer1.add_scalar('a1', a1, global_step=num)
            writer1.add_scalar('a2', a2, global_step=num)
            writer1.add_scalar('a3', a3, global_step=num)

            writer1.add_images('label', label, global_step=num)
            su += a3.item()
            print(su / num)
            # scaled_disp, _ = disp_to_depth(disp, 0.1, 10)
            # Saving colormapped depth image
            # vmax = np.percentile(disp_resized_np, 95)
        writer1.close()
        print('-> Done!')

def label2target(label):
    targets = {}
    mask = torch.ones(label.shape)
    targets['gt_mask'] = mask.detach().numpy()
    targets['depth_gt'] = label.detch().numpy()
    return targets



if __name__ == '__main__':
    import sys
    sys.path.append("..") # 这句是为了导入_config
    args = parse_args()
    #inference(args)
    mark_pru()
