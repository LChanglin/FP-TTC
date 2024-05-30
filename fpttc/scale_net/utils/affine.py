import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def affine(pref,flow, pw=2):
    b,_,lh,lw=flow.shape
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
    return torch.reciprocal(exp), Error, mask

def affine_x_y(flow):
    b, _, h, w = flow.size()
    grid_H = torch.linspace(0, w-1, w).view(1, 1, 1, w).expand(b, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    grid_V = torch.linspace(0, h-1, h).view(1, 1, h, 1).expand(b, 1, h, w).to(device=flow.device, dtype=flow.dtype)
    grids1_ = torch.cat([grid_H, grid_V], dim=1)


    gw = 1
    pad_dim = (gw,gw,gw,gw)
    grids_pad = F.pad(grids1_, pad_dim, "replicate")
    flow_pad = F.pad(flow, pad_dim, "replicate")
    grids_w_f = grids_pad + flow_pad

    # tm - m
    len_ori = torch.abs(grids_pad[...,0:-2*gw,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,0:-2*gw,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_f_len = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    exp_y_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # bm - m
    len_ori = torch.abs(grids_pad[...,2*gw:,gw:-gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,2*gw:,gw:-gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_f_len += (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    exp_y_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    exp_y_len = torch.min(exp_y_len1, exp_y_len2)

    # mr - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,2*gw:] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,2*gw:] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_f_len += (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    exp_x_len1 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    # ml - m
    len_ori = torch.abs(grids_pad[...,gw:-gw,0:-2*gw] - grids_pad[...,gw:-gw,gw:-gw]) 
    len_scale_f = torch.abs(grids_w_f[...,gw:-gw,0:-2*gw] - grids_w_f[...,gw:-gw,gw:-gw]) 
    len_ori_x, len_ori_y = torch.abs(len_ori[:,0:1,...]), torch.abs(len_ori[:,1:,...])
    len_scale_f_x, len_scale_f_y = torch.abs(len_scale_f[:,0:1,...]), torch.abs(len_scale_f[:,1:,...])
    exp_f_len += (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)
    exp_x_len2 = (len_ori_x**2+len_ori_y**2)**0.5 * torch.reciprocal((len_scale_f_x**2+len_scale_f_y**2)**0.5)

    exp_x_len = torch.min(exp_x_len1, exp_x_len2)

    exp = torch.max(torch.stack([exp_x_len,exp_y_len],dim=1),dim=1)[0]
    exp = exp.clamp(0.5,2)

    return exp