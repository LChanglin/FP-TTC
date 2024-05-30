from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

def get_loss(scale, flow, dc_gt, flow_gt, valid, epoch):

    gt_dchange = dc_gt[:,0:1,:,:]
    valids = dc_gt[:, 1, :, :].unsqueeze(1).bool()

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 400)

    gt_dchange[gt_dchange<=0] = 10
    gt_dchange[gt_dchange>3] = 10

    maskdc = ((gt_dchange < 3) & (gt_dchange > 0.3) & valids & (scale>0))

    d_loss = (scale.log() - gt_dchange.log()).abs()
    f_loss = (flow-flow_gt).abs()
    #print(scale.log(), gt_dchange.log())
    sloss = (maskdc * d_loss).sum()/maskdc.sum()
    floss = (valid[:, None] * f_loss).sum()/valid.sum()

    loss = 0.8*sloss + 0.2*floss

    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid.view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    f1 = torch.mean(1-out[val])

    return sloss, loss, gt_dchange, f1


def get_loss_multi(scale_preds, valid, scale_gt, epoch, gamma=0.9):

    n_predictions = len(scale_preds)
    scale_loss = 0.0

    gt_depth = scale_gt[:,0:1,:,:]
    gt_depth[gt_depth<=0] = 1e6
    gt_f3d =  scale_gt[:,1:,...].clone()
    gt_dchange = (1+gt_f3d[:,2:,...]/gt_depth)

    validx = ((gt_dchange < 3) & (gt_dchange > 0.2) & valid.unsqueeze(1).bool())

    for i in range(n_predictions):
        scale = scale_preds[i]
        i_weight = gamma ** (n_predictions - i - 1)
        # if i==0:
        #     first = n_predictions//2
        #     #print(first)
        #     i_weight = gamma ** (n_predictions - first)
        # else:
        #     i_weight = gamma ** (n_predictions - i - 1)

        mask_nan = ~torch.isnan(scale)
        maskdc = validx & mask_nan
        mask_minus0 = (scale<=0)

        if mask_minus0.sum() == 0:    
            loss1 =  ((((scale.abs()).log()-(gt_dchange.abs()).log()).abs())*maskdc).sum() / (maskdc.sum())
            loss = loss1
            if epoch < 5:
                loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
                loss += loss2
        else:
            loss1 = (((scale-gt_dchange).abs())[mask_nan]).mean()
            loss2 = (((scale.abs())*mask_minus0)).sum() / (mask_minus0.sum()+1e-4)
            loss = loss1 + loss2

        scale_loss += i_weight * loss.mean()
        if i == n_predictions - 1:
            loss_final = loss.mean()

    return scale_loss, loss_final, gt_dchange, maskdc

def ttc_smooth_loss(img, disp, mask):
    """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
    """
    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp - torch.roll(disp, 1, dims=3))
    grad_disp_y = torch.abs(disp - torch.roll(disp, 1, dims=2))
    grad_disp_x[:,:,:,0] = 0
    grad_disp_y[:,:,0,:] = 0

    # grad_disp_xx = torch.abs(torch.roll(grad_disp_x, -1, dims=3) - grad_disp_x)
    # grad_disp_yy = torch.abs(torch.roll(grad_disp_y, -1, dims=3) - grad_disp_y)
    # grad_disp_xx[:,:,:,0] = 0
    # grad_disp_yy[:,:,0,:] = 0
    # grad_disp_xx[:,:,:,-1] = 0
    # grad_disp_yy[:,:,-1,:] = 0

    grad_img_x = torch.mean(torch.abs(img - torch.roll(img, 1, dims=3)), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img - torch.roll(img, 1, dims=2)), 1, keepdim=True)
    grad_img_x[:,:,:,0] = 0
    grad_img_y[:,:,0,:] = 0

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return (grad_disp_x*mask).sum()/mask.sum() + (grad_disp_y*mask).sum()/mask.sum()





def get_loss_selfsup(scale, valid, scale_gt):
    return (scale.log()-scale_gt.log()).abs()[valid.bool()].mean()


def self_supervised_gt_affine(flow):

    b,_,lh,lw=flow.shape
    bs, w,h = b, lw, lh
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(bs, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    pref = torch.cat([grid_H, grid_V], dim=1)
    ptar = pref + flow
    pw = 1
    pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
    ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
    pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
    ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

    prefprefT = pref.matmul(pref.permute(0,2,1))
    ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
    ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

    Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
    Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

    Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
    exp = Avol.sqrt()
    mask = (exp>0.5) & (exp<2)
    mask = mask[:,0]

    exp = exp.clamp(0.5,2)
    # exp[Error>0.1]=1
    return torch.reciprocal(exp)


def self_supervised_gt(flow_f):

    b,_,h,w = flow_f.size()
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=flow_f.device, dtype=flow_f.dtype)
    grids1_ = torch.cat([grid_H, grid_V], dim=1)

    gw = 2
    pad_dim = (gw,gw,gw,gw)
    grids_pad = F.pad(grids1_, pad_dim, "replicate")
    flow_f_pad = F.pad(flow_f, pad_dim, "replicate")
    grids_w_f = grids_pad + flow_f_pad

    # tm - m
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # bm - m
    len_ori = torch.abs(grids_pad[...,2*gw:,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,2*gw:,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_y_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # mr - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_x_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - mr
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,2*gw:]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,2*gw:]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # ml - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_l_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # tm - ml
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,0:-2*gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,0:-2*gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    # mr - bm
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,2*gw:,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,2*gw:,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_r_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)


    exp_x_len = torch.min(torch.stack([exp_x_len1,exp_x_len2],dim=1),dim=1)[0]
    exp_y_len = torch.min(torch.stack([exp_y_len1,exp_y_len2],dim=1),dim=1)[0]
    exp_l_len = torch.min(torch.stack([exp_l_len1,exp_l_len2],dim=1),dim=1)[0]
    exp_r_len = torch.min(torch.stack([exp_r_len1,exp_r_len2],dim=1),dim=1)[0]

    exp_corner_len = torch.max(torch.stack([exp_l_len,exp_r_len],dim=1),dim=1)[0]
    exp_f_xy_len = torch.max(exp_x_len,exp_y_len)

    threshold = 0.15
    ttc_range = 0.95
    mask_exp1 = torch.logical_and(exp_corner_len<1. ,(exp_corner_len/exp_f_xy_len)>1)
    mask_exp2 = torch.logical_and((exp_x_len/exp_y_len)>(1-threshold),(exp_x_len/exp_y_len)<(1+threshold))
    mask_exp2 = torch.logical_and(mask_exp2,exp_f_xy_len<ttc_range*torch.mean(exp_f_xy_len))
    mask_exp = torch.logical_and(mask_exp1, mask_exp2)

    scale_gt = mask_exp*exp_corner_len+ ~mask_exp*exp_f_xy_len
    scale_gt[...,0:1,:] = 1
    scale_gt[...,-1:,:] = 1
    scale_gt[...,:,0:1] = 1
    scale_gt[...,:,-1:] = 1

    return scale_gt

