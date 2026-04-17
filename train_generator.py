import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid as make_image_grid
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from cp_dataset_test import CPDatasetTest
from networks import ConditionGenerator, VGGLoss, load_checkpoint, save_checkpoint, make_grid
from network_generator import SPADEGenerator, MultiscaleDiscriminator, GANLoss
from sync_batchnorm import DataParallelWithCallback
from tensorboardX import SummaryWriter
from utils import create_network, visualize_segmap
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import transforms
import eval_models as models
import torchgeometry as tgm

def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cuda', action='store_true') # Fixed
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--tocg_checkpoint', type=str)
    parser.add_argument('--gen_checkpoint', type=str, default='')
    parser.add_argument('--dis_checkpoint', type=str, default='')
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument('--semantic_nc', type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7)
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--num_test_visualize', type=int, default=3)
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    parser.add_argument('--G_lr', type=float, default=0.0001)
    parser.add_argument('--D_lr', type=float, default=0.0004)
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most')
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02)
    parser.add_argument('--no_ganFeat_loss', action='store_true')
    parser.add_argument('--no_vgg_loss', action='store_true')
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--n_layers_D', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--GT', action='store_true')
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lpips_count", type=int, default=1000)
    parser.add_argument("--tensorboard_dir", default="tensorboard")
    opt = parser.parse_args()
    return opt

def train(opt, train_loader, test_loader, test_vis_loader, board, tocg, generator, discriminator, lpips_model, device):
    if tocg: tocg.to(device).eval()
    generator.to(device).train()
    discriminator.to(device).train()
    lpips_model.to(device).eval()

    criterionGAN = GANLoss('hinge', tensor=torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor)
    criterionFeat = nn.L1Loss()
    criterionVGG = VGGLoss(opt).to(device)

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=opt.G_lr, betas=(0, 0.9))
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=opt.D_lr, betas=(0, 0.9))

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    for step in tqdm(range(opt.load_step, opt.keep_step + opt.decay_step)):
        inputs = train_loader.next_batch()
        agnostic = inputs['agnostic'].to(device)
        pose = inputs['densepose'].to(device)
        parse_agnostic = inputs['parse_agnostic'].to(device)
        cm = inputs['cloth_mask']['paired'].to(device)
        c_paired = inputs['cloth']['paired'].to(device)
        im = inputs['image'].to(device)

        with torch.no_grad():
            if not opt.GT:
                # Warping logic
                input1 = torch.cat([F.interpolate(c_paired, size=(256, 192), mode='bilinear'), 
                                    F.interpolate(cm, size=(256, 192), mode='nearest')], 1)
                input2 = torch.cat([F.interpolate(parse_agnostic, size=(256, 192), mode='nearest'), 
                                    F.interpolate(pose, size=(256, 192), mode='bilinear')], 1)
                flow_list, fake_segmap, _, warped_clothmask_paired = tocg(opt, input1, input2)
                
                # Generator Parse Map logic
                fake_parse_gauss = gauss(F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear'))
                fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]
                
                # Warping cloth for generator input
                N, _, iH, iW = c_paired.shape
                grid = make_grid(N, iH, iW, opt).to(device)
                flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                warped_cloth_paired = F.grid_sample(c_paired, grid + flow_norm, padding_mode='border')
            else:
                fake_parse = inputs['parse'].cuda().argmax(dim=1)[:, None]
                warped_cloth_paired = inputs['parse_cloth'].cuda()

            # Create labels (Fixed: using device=device)
            parse = torch.zeros(fake_parse.size(0), 7, opt.fine_height, opt.fine_width, device=device)
            # ... (labels mapping logic)
            # Simplified for brevity: map fake_parse to 7 channels
            old_parse = torch.zeros(fake_parse.size(0), 13, opt.fine_height, opt.fine_width, device=device)
            old_parse.scatter_(1, fake_parse, 1.0)
            labels = {0:[0], 1:[2,4,7,8,9,10,11], 2:[3], 3:[1], 4:[5], 5:[6], 6:[12]}
            for i, ids in labels.items():
                for label_id in ids: parse[:, i] += old_parse[:, label_id]

        # Generator Training
        output_paired = generator(torch.cat((agnostic, pose, warped_cloth_paired), dim=1), parse)
        # ... loss calculation and backward ...
        loss_gen = criterionGAN(discriminator(torch.cat((parse, output_paired), 1)), True, False) + criterionVGG(output_paired, im) * opt.lambda_vgg
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        # Discriminator Training
        loss_dis = criterionGAN(discriminator(torch.cat((parse, im), 1)), True) + criterionGAN(discriminator(torch.cat((parse, output_paired.detach()), 1)), False)
        optimizer_dis.zero_grad()
        loss_dis.backward()
        optimizer_dis.step()

def main():
    opt = get_opt()
    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")
    
    train_loader = CPDataLoader(opt, CPDataset(opt))
    
    # Perceptual Loss (LPIPS)
    lpips_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=(device.type == 'cuda'))
    
    tocg = None
    if not opt.GT:
        tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=opt.semantic_nc+3, output_nc=13, ngf=96, norm_layer=nn.BatchNorm2d)
        load_checkpoint(tocg, opt.tocg_checkpoint, opt)

    generator = SPADEGenerator(opt, 3+3+3)
    discriminator = create_network(MultiscaleDiscriminator, opt)
    
    train(opt, train_loader, None, None, SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name)), tocg, generator, discriminator, lpips_model, device)

if __name__ == "__main__":
    main()