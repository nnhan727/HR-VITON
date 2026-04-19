import torch
import torch.nn as nn
from torchvision.utils import make_grid
from networks import make_grid as mkgrid
import argparse
import os
import time
from cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import Subset

def iou_metric(y_pred_batch, y_true_batch):
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        y_pred = y_pred > 0.5
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)
        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou

def remove_overlap(seg_out, warped_cm):
    assert len(warped_cm.shape) == 4
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint')
    parser.add_argument('--tocg_checkpoint', type=str, default='')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)

    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true')  
    parser.add_argument('--Ddropout', action='store_true')
    parser.add_argument('--num_D', type=int, default=2)

    parser.add_argument('--cuda', action='store_true', help='use cuda')

    parser.add_argument("--G_D_seperate", action='store_true')
    parser.add_argument("--no_GAN_loss", action='store_true')
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true')
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge")
    parser.add_argument('--add_lasttv', action='store_true')
    
    parser.add_argument("--no_test_visualize", action='store_true')    
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")

    parser.add_argument('--G_lr', type=float, default=0.0002)
    parser.add_argument('--D_lr', type=float, default=0.0002)
    parser.add_argument('--CElamda', type=float, default=10)
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--val_count', type=int, default=1000)
    parser.add_argument('--spectral', action='store_true')
    parser.add_argument('--occlusion', action='store_true')
    
    opt = parser.parse_args()
    return opt

def train(opt, train_loader, test_loader, val_loader, board, tocg, D, device):
    tocg.to(device)
    D.to(device)
    tocg.train()
    D.train()

    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt).to(device) # Fixed: Move to device
    
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor)

    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))

    for step in tqdm(range(opt.load_step, opt.keep_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c_paired = inputs['cloth']['paired'].to(device)
        cm_paired = inputs['cloth_mask']['paired'].to(device)
        cm_paired = (cm_paired > 0.5).float()
        
        parse_agnostic = inputs['parse_agnostic'].to(device)
        densepose = inputs['densepose'].to(device)
        openpose = inputs['pose'].to(device)
        label_onehot = inputs['parse_onehot'].to(device)
        label = inputs['parse'].to(device)
        parse_cloth_mask = inputs['pcm'].to(device)
        im_c = inputs['parse_cloth'].to(device)
        im = inputs['image'].to(device)

        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
        
        warped_cm_onehot = (warped_clothmask_paired.detach() > 0.5).float()
        
        if opt.clothmask_composition != 'no_composition':
            cloth_mask = torch.ones_like(fake_segmap.detach())
            if opt.clothmask_composition == 'detach':
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
            else: # warp_grad
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
            fake_segmap = fake_segmap * cloth_mask

        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)
        
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = (fake_clothmask.float() - warped_cm_onehot).clamp(min=0.0)
        
        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        # ... (TV loss logic remains same, just ensure flow is on device)
        for flow in (flow_list if not opt.lasttvonly else flow_list[-1:]):
            y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
            x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
            loss_tv = loss_tv + y_tv + x_tv

        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        
        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)
            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            loss_G_GAN = criterionGAN(pred_segmap, True)
            
            # Combined Loss
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Discriminator Step
            real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label), dim=1))
            fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()), dim=1))
            loss_D = criterionGAN(real_segmap_pred, True) + criterionGAN(fake_segmap_pred, False)
            
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_step_%06d.pth' % (step + 1)), opt)

def main():
    opt = get_opt()
    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")
    
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    # Test loader setup...
    test_loader = None
    val_loader = None
    if not opt.no_test_visualize:
        test_dataset = CPDatasetTest(opt)
        val_dataset = Subset(test_dataset, np.arange(min(2000, len(test_dataset))))
        test_loader = CPDataLoader(opt, test_dataset)
        val_loader = CPDataLoader(opt, val_dataset)

    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    input1_nc = 4
    input2_nc = opt.semantic_nc + 3
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2=opt.Ddownx2, Ddropout=opt.Ddropout, n_layers_D=3, spectral=opt.spectral, num_D=opt.num_D)
    
    if opt.tocg_checkpoint and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint, opt)

    train(opt, train_loader, test_loader, val_loader, board, tocg, D, device)

if __name__ == "__main__":
    main()