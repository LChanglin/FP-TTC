import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.matching import (global_correlation_softmax, local_correlation_softmax, local_correlation_with_flow)
from ..modules.attention import SelfAttnPropagation
from ..modules.utils import upsample_flow_with_mask
from ..modules.reg_refine import BasicUpdateBlock


class FlowNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=8,
                 reg_refine=False,  # optional local regression refinement
                 ):
        super(FlowNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine


        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)

        if not self.reg_refine:
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

        if self.reg_refine:
            self.refine_proj = nn.Conv2d(128, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2,
                                           bilinear_up=False,
                                           )


    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow


    def forward(self, feature0, feature1,
                scale_idx,
                corr_radius_list=None,
                prop_radius_list=None,
                num_reg_refine=6,
                pred_bidir_flow=False,
                flow = None,
                ):

        flow_preds = []
        mlvl_flows = []

        feature0_ori, feature1_ori = feature0, feature1

        upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

        corr_radius = corr_radius_list[scale_idx]
        prop_radius = prop_radius_list[scale_idx]


        # correlation and softmax
        if corr_radius == -1:  # global matching
            flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
        else:  # local matching
            flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]

        # flow or residual flow
        flow = flow + flow_pred if flow is not None else flow_pred


        flow = self.feature_flow_attn(feature0, flow.detach(),
                                        local_window_attn=prop_radius > 0,
                                        local_window_radius=prop_radius,
                                        )
        if scale_idx < self.num_scales - 1:
            return flow, None

        # flow = flow.detach()
        # self.eval()
        # torch.set_grad_enabled(False)
        flow_up = self.upsample_flow(flow, feature0)
        # torch.set_grad_enabled(True)
        # self.train()

        return flow, flow_up
