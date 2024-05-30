import os
import torch
import torchvision.transforms as transforms
import cv2
import sys
import numpy as np
import datetime
import random
import math
import  time
import torch.optim as optim
import torch.nn as nn
from random import sample, shuffle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F 
import torch.distributed as dist

from PIL import Image

from .loss import get_loss, get_loss_multi
from .draw import disp2rgb, flow_uv_to_colors, flow_to_image

from dataloader.load import load_calib_cam_to_cam, readFlowKITTI, disparity_loader, triangulation



class TTCTrainer(object):
    def __init__(self, model, dataset, optimizer, args, device, model_path = None, 
                start_epoch=0, parallel=False, time_stamp=None, max_lr=1e-4, crop_size=[352,1152]):
        self.model = model
        self.parallel = parallel
        self.batch_size = args.batch_size
        self.parallel = parallel
        self.train_sampler = None
        if not self.parallel:
            self.train_loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=True, 
                            num_workers=args.batch_size, drop_last=True, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(dataset)
            self.train_loader = DataLoader(dataset,batch_size=args.batch_size, \
                                sampler=self.train_sampler, shuffle=False, pin_memory=True, num_workers=2)

        self.epoch = args.epoch
        # if optimizer is None:
        #     self.optimizer = self.get_optimizer()
        # else :
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        
        steps_per_epoch = int(len(self.train_loader))
        #print(self.epoch, len(self.train_loader), self.batch_size, steps_per_epoch)
        starte = -1
        if self.start_epoch>0:
            starte = self.start_epoch - 1
        self.lr_scheduler3 = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr,
            epochs=self.epoch,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=max(steps_per_epoch*starte,-1),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
                        milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110,\
                        115, 120, 125, 130, 135, 140, 145, 160, 170, 180, 190, 200, 210], gamma=0.5, last_epoch=max((self.start_epoch-1),-1))
        self.lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
                        milestones=[200], gamma=2, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 120, 125, 130], gamma=0.5, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[3,6,9,12,15,18,21,24,27,30,33,36,39,42,57,60,63,66,69,86,89,92,95], gamma=0.75, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[5,10,15,20,25,30,35,40,45,55,65,75,85], gamma=0.5, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[10,20,30,40,50,60,70,75,80,85,90,95,100,110,115,120], gamma=0.5, last_epoch=max((self.start_epoch-1),-1))
        # self.lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, \
        #                 milestones=[5,10,15,20,25,30,35,40,50,60,70,80,90,100,110,115,120], gamma=0.6, last_epoch=max((self.start_epoch-1),-1))
    
        self.device = device
        self.train_loss_history = []
        self.plt_train_epoch = []
        if time_stamp is None:
            self.time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
            out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
        else:
            self.time_stamp = time_stamp
        self.model_path = model_path
        
        self.attn_type=args.attn_type
        self.attn_splits_list=args.attn_splits_list
        self.corr_radius_list=args.corr_radius_list
        self.prop_radius_list=args.prop_radius_list
        self.num_reg_refine=args.num_reg_refine

        self.grad_clip = 1.0
        self.checkpoint = 1

        self.loss_per_epoch = 0
        self.loss_sum_per_epoch = 0
        self.iters = 0

        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': True}

    def get_optimizer(self):
        params = list(self.model.named_parameters())
        param_group = [
            {'params':[p for n,p in params if 'featnet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'flownet' in n],'lr':1e-5},
            {'params':[p for n,p in params if 'scalenet' in n],'lr':1e-4},
        ]
        optimizer = torch.optim.Adam(param_group,lr=1e-5)
        return optimizer

    def train(self):
        start = time.time()
        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
        loss_txt = out_dir + '/0.txt'
        # file = open(loss_txt,'w')
        # file.close()
        #self.lr_scheduler2.step()
        print("Learning rate: ", self.optimizer.state_dict()['param_groups'][0]['lr'])
        # loss_now = self.eval_epoch()
        # print('Eval: ', loss_now)
        for epoch in range(self.start_epoch, self.epoch):
            print('Epoch:', epoch)
            self.loss_per_epoch = 0
            self.loss_sum_per_epoch = 0
            self.iters = 0
            # loss_now
            # if torch.distributed.get_rank()==0:
            #print(('Train %f %f '%(self.loss_per_epoch/max(1,self.iters), self.loss_sum_per_epoch/max(1,self.iters))))
            self.train_epoch(epoch)
            loss_now = self.eval_epoch()
            print('Eval: ', loss_now)
            if (self.parallel and torch.distributed.get_rank()==0) or (not self.parallel):
                file = open(loss_txt,'a')
                file.write('Loss in epoch %d: '%(epoch))
                # file.write('Train %f %f '%(self.loss_per_epoch/max(1,self.iters), self.loss_sum_per_epoch/max(1,self.iters))) 
                file.write('Train %f'%(self.loss_per_epoch/max(1,self.iters)))   
                file.write('Test %f\n'%(loss_now))          
                file.close()

                if (epoch<self.epoch and epoch%self.checkpoint==0):
                    checkpoint = {
                            "net": self.model.state_dict(),
                            'optimizer':self.optimizer.state_dict(),
                            "epoch": epoch+1
                        }
                    temp_pth = self.model_path[:-8] + str(epoch) + 'y.pth.tar'
                    # temp_pth = self.model_path[:-10] + '_y' + str(epoch) + '.pth.tar'
                    torch.save(checkpoint, temp_pth)
                
                print("Loss in epoch", epoch, ":", self.loss_per_epoch/max(1,self.iters))
                print("Learning rate: ", self.optimizer.state_dict()['param_groups'][0]['lr'])


        # end = time.time()
        # total_time = end-start
        # print(f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s')
        # torch.save(self.model.state_dict(), self.model_path)
    
    def train_epoch(self, epoch):
        total_samples = len(self.train_loader.dataset)
        if self.parallel:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        ten_loss_sum = 0

        out_dir = "./log/%s_selfcon_ttc"%(self.time_stamp)
        save_index = 2000 if not self.parallel else random.randint(int(4000/self.batch_size),int(8000/self.batch_size))
        t0 = time.time()
        for i, data in enumerate(self.train_loader):
            
            #print('aaaaaaaaaaaaaaaaaa', time.time()-t0)
            #print('xxxxxxxxxxxxxxxxxx', i)
            img0, img1, flow_gt, imgAux , valid = data
            img0, img1 = img0.to(self.device), img1.to(self.device)
            # flow_gt ,imgAux = flow_gt.to(self.device), imgAux.to(self.device)
            imgAux = imgAux.to(self.device)
            valid = valid.to(self.device)
            flow_gt = flow_gt.to(self.device)
            start_time = time.time()
            #print(img0.shape)
            self.optimizer.zero_grad()
            # for param in self.model.parameters():
            #     param.grad = None
            
            time2 = time.time()

            scale, flow, scale_pre = self.model(img0, img1,
                                    attn_type=self.attn_type,
                                    attn_splits_list=self.attn_splits_list,
                                    corr_radius_list=self.corr_radius_list,
                                    prop_radius_list=self.prop_radius_list,
                                    num_reg_refine=self.num_reg_refine,
                                    )

            end_time = time.time()
            #print(end_time-time2)
            # print(scale.shape, flow.shape)
            # scale_gt_selfsup = self_supervised_gt_affine(flow)[0]
            # scale_gt_selfsup[~(valid[0].unsqueeze(0).bool())] = 0
            if type(scale) == list:
                loss, loss_last, gt_scale, mask = get_loss_multi(scale, imgAux, epoch)
            else:
                loss, loss_last, gt_scale, f1 = get_loss(scale, flow, imgAux, flow_gt, valid, epoch)

            # torch.distributed.barrier()
            time3 = time.time()
            # loss += ttc_smooth_loss(img1, scale, torch.ones_like(valid))
            # if isinstance(loss, float):
            #     continue

            # if torch.isnan(loss):
            #     continue
            #print(scale_gt_selfsup.shape, gt_scale.shape)
            loss_last.backward()
            #self.average_gradients()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            self.lr_scheduler3.step()
            #print(time.time()-time3, time3-time2, time2-time1)

            #print(flow.shape, scale.shape, gt_scale.shape, img0.shape)
            if type(scale) == list:
                scale = scale[-1]
            if i%int(save_index)==0:
                out_of_viz = flow_to_image((flow.permute(0,2,3,1).detach().cpu().numpy())[0])
                cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'flow'+'.jpg'), out_of_viz)

                out_of_viz = flow_to_image((flow_gt.permute(0,2,3,1).detach().cpu().numpy())[0])
                cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'flowgt'+'.jpg'), out_of_viz)

                ttc_warp_image2 = ((scale[0]).transpose(0,1).transpose(1,2) - 0.5) / (1.0)
                ttc_warp_image2 = disp2rgb(np.clip(ttc_warp_image2.detach().cpu().numpy(), 0.0, 1.0))
                ttc_warp_image2 = ttc_warp_image2*255.0
                cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'ttcs'+'.jpg'), ttc_warp_image2)

                # img = ((img0[0]).transpose(0,1).transpose(1,2))
                # img = np.clip(img.detach().cpu().numpy(), 0.0, 255.0)
                # cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'img'+'.jpg'), img)
                
                img = ((img0[0]).transpose(0,1).transpose(1,2))
                img = np.clip(img.detach().cpu().numpy(), 0.0, 255.0)
                cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'img1'+'.jpg'), img)

                # ttc_warp_image = ((scale_gt_selfsup[0:1,...]).transpose(0,1).transpose(1,2) - 0.5) / (1.0)
                # ttc_warp_image = disp2rgb(np.clip(ttc_warp_image.detach().cpu().numpy(), 0.0, 1.0))
                # ttc_warp_image = ttc_warp_image*255.0
                # cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'ttc_self'+'.jpg'), ttc_warp_image)

                # occ_image = (mask[0]).transpose(0,1).transpose(1,2).detach().cpu().numpy()
                # occ_image = (occ_image*255)
                # occ_image = np.repeat(occ_image, 3, axis=2)
                # occ_image = np.asarray(occ_image, dtype=np.uint8)
                # cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'mask'+'.jpg'), occ_image)

                ttc_warp_image = ((gt_scale[0,...]).transpose(0,1).transpose(1,2) - 0.5) / (1.0)
                ttc_warp_image = disp2rgb(np.clip(ttc_warp_image.detach().cpu().numpy(), 0.0, 1.0))
                ttc_warp_image = ttc_warp_image*255.0
                cv2.imwrite(os.path.join(out_dir, str(epoch)+'_'+str(i)+'ttcx'+'.jpg'), ttc_warp_image)


            # if (self.parallel and torch.distributed.get_rank()==0):
            #     dist.all_reduce(loss, op = dist.ReduceOp.SUM)
            #     loss /= float(dist.get_world_size())
            #     if i % 10 == 0:
            #         print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
            #             ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
            #             '{:6.4f}'.format(loss.item()))
            # else:
            if loss_last is not None:
                self.loss_per_epoch += loss.item()
                self.loss_sum_per_epoch += loss_last.item()
                if i % 10 == 0:
                        print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
                            ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
                            '{:6.4f}'.format(loss_last.item()) + '    ' + '{:6.4f}'.format(loss.item()), '    ' + '{:6.4f}'.format(f1.item()))
            else:
                self.loss_per_epoch += loss.item()
                if i % 10 == 0:
                    print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
                        ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
                        '{:6.4f}'.format(loss.item()))

                                   
            # if i % 10 == 0:
            #     print('[' +  '{:5}'.format(i *self.train_loader.batch_size) + '/' + '{:5}'.format(total_samples) +
            #         ' (' + '{:3.0f}'.format(100 * i / len(self.train_loader)) + '%)]  Loss_now: ' +
            #         '{:6.4f}'.format(loss.item()) + '   Loss eva: ' + '{:6.4f}'.format(loss2.item()))

            # self.loss_per_epoch += loss2.item()
            self.iters += 1

            t0 = time.time()
                
            #print('Epoch Time: ', end_time-start_time)
        
    @torch.no_grad()
    def eval_epoch(self, eval_path='/mnt/pool/lcl/data/kitti/data_scene_flow/training/'):

        img0x, img1x, flow0 = self.eva_dataloader(eval_path)
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flow0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flow0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flow0]

        #w0,h0 = 1152,320
        w0,h0 = 960,288

        total_loss=0
        t=0

        self.model.eval()

        for i in range(len(img0x)):
            # print(flow0[i])
            flow, valid = readFlowKITTI(flow0[i])
            ints = load_calib_cam_to_cam(calib[i])
            fl = ints['K_cam2'][0,0]
            cx = ints['K_cam2'][0,2]
            cy = ints['K_cam2'][1,2]
            bl = ints['b20']-ints['b30']
            d1 = disparity_loader(disp0[i])
            d2 = disparity_loader(disp1[i])

            flow = np.ascontiguousarray(flow,dtype=np.float32)
            flow[np.isnan(flow)] = 1e6 # set to max
            valid = np.logical_and(np.logical_and(valid>0.99, d1>0), d2>0)
            d1[d1<=0] = 1e6
            d2[d2<=0] = 1e6

            shape = d1.shape
            mesh = np.meshgrid(range(shape[1]),range(shape[0]))
            xcoord = mesh[0].astype(float)
            ycoord = mesh[1].astype(float)
            
            # triangulation in two frames
            P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl = fl, cx = cx, cy = cy)
            P1 = triangulation(d2, xcoord + flow[:,:,0], ycoord + flow[:,:,1], bl=bl, fl = fl, cx = cx, cy = cy)
            dis0 = P0[2]
            dis1 = P1[2]

            change_size =  dis0.reshape(shape).astype(np.float32)
            valid = np.logical_and(valid, change_size>0).astype(float)
            flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))
            change_size = np.concatenate((change_size[:,:,np.newaxis],flow3d),2)
            scale_gt = (np.array(change_size).astype(np.float32))

            gt_depth = scale_gt[...,0:1]
            # gt_depth[gt_depth<=0] = 1e6
            gt_f3d =  scale_gt[...,1:]
            gt_dchange = (1+gt_f3d[...,2:]/gt_depth)
            maskdc = (gt_dchange < 3) & (gt_dchange > 0.3) & np.expand_dims(valid.astype(bool),axis=2)


            file_1 = img0x[i]
            file_2 = img1x[i]

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)          
            image1 = cv2.resize(image1, (w0,h0))
            image2 = cv2.resize(image2, (w0,h0))
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)


            padding_factor = 32
            inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

            ori_size = image1.shape[-2:]
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                        align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                        align_corners=True)
            scale, flow, _ = self.model(image1, image2,
                                    attn_type=self.attn_type,
                                    attn_splits_list=self.attn_splits_list,
                                    corr_radius_list=self.corr_radius_list,
                                    prop_radius_list=self.prop_radius_list,
                                    num_reg_refine=self.num_reg_refine,
                                    testing=True,
                                    )
            if type(scale) == list:
                scale = scale[-1]
            flow_pr = scale
            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                        align_corners=True)
                flow_pr = flow_pr * ori_size[-1] / inference_size[-1]
            
            scale = flow_pr[0,0].detach().cpu().numpy()
            h,w,_ = gt_dchange.shape
            if scale.shape[0]!=h and scale.shape[1]!=w:
                scale = cv2.resize(scale, (w,h))
            scale = np.expand_dims(scale, axis=2)
            loss =  (np.abs((np.log(scale)-np.log(gt_dchange))))[maskdc].mean()
            total_loss += loss.mean()
            t += 1

        loss = total_loss / float(t)

        return loss

    def eva_dataloader(self, filepath):

        img_file_path = filepath
        left_fold  = 'image_2/'
        flow_noc   = 'flow_occ/'

        train_img = [img for img in os.listdir(img_file_path+left_fold) if img.find('_10') > -1]
        train_img = [i for i in train_img if int(i.split('_')[0])%5==0]
        train = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
        train = [i for i in train if int(i.split('_')[0])%5==0]

        l0_train  = [img_file_path+left_fold+img for img in train_img]
        l1_train = [img_file_path+left_fold+img.replace('_10','_11') for img in train_img]
        flow_train = [filepath+flow_noc+img for img in train]

        return sorted(l0_train), sorted(l1_train), sorted(flow_train)

    def average_gradients(self):  ##每个gpu上的梯度求平均
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data,op = dist.reduce_op.SUM)
                param.grad.data /= size
