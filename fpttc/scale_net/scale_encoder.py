import math
import time
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init

from .utils.affine import affine, affine_x_y
from ..modules.position import PositionEmbeddingSine
from ..modules.geometry import flow_wrap
from .utils.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from ..modules.attention import SelfAttnPropagation


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )


class ScaleEncoder(nn.Module):
    def __init__(self,
                 num_layers=2,
                 input_dim=128,
                 d_model=128,
                 nhead=4,
                 num_feature_levels=2,
                 num_level=4,
                 ):
        super(ScaleEncoder, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.embed_dims = input_dim
        self.num_level = num_level



        # self.feature2scale_proj = nn.Linear(d_model, 1, bias=False)

        # self.scale_conv = nn.Conv2d(1, d_model, 3, padding=1)
        self.scale_conv = nn.Sequential(
            conv(input_dim, d_model*2, padding_mode="zeros"),
            conv(d_model*2, d_model, padding_mode="zeros"),
            conv(d_model, d_model, padding_mode="zeros")
        )
        self.query_conv1 = conv(2*d_model, d_model)
        self.query_conv2 = conv(d_model, d_model)

        self.value_fs_conv = nn.Conv1d(2*input_dim, d_model, 1)

        self.pos_enc = PositionEmbeddingSine(num_pos_feats=d_model/2)

        # print('xxx', nhead)
        self.layers = nn.ModuleList([
            TransformerBlock(num_level=num_level,
                             d_model=d_model,
                             num_head=nhead,
                             )
            for i in range(num_layers)])

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    '''
    query: scale0 + scale1
    value: feats0 + feats1
    '''
    def forward(self, feature0_list, feature1_list, 
                query_lvl = -1, ini_query=None
                ):
        
        assert len(feature0_list)==self.num_level
        bs, _, height, width = feature1_list[query_lvl].shape

        # query

        scale_pre = feature0_list[query_lvl]

        if ini_query is None:
            query = self.scale_conv(scale_pre)  # b, c, h, w
        else:
            query = self.scale_conv(ini_query)
        query += self.pos_enc(query)    # b, c, h, w

        query_location = self.get_reference_points(height, width, bs, device=query.device, dtype=query.dtype)

        # value
        feat0_flatten = []
        feat1_flatten = []
        spatial_shapes = []

        t0 = time.time()
        for lvl, feat in enumerate(feature0_list):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2) # (bs, c, H*W)
            feat = feat + self.level_embeds[None, lvl:lvl + 1, :].permute(0,2,1).contiguous().to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat0_flatten.append(feat)

        for lvl, feat in enumerate(feature1_list):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2) # (bs, c, H*W)
            feat = feat + self.level_embeds[None, lvl:lvl + 1, :].permute(0,2,1).contiguous().to(feat.dtype)
            feat1_flatten.append(feat)

        feat0_flatten = torch.cat(feat0_flatten, 2)   #b, c, hw+HW
        feat1_flatten = torch.cat(feat1_flatten, 2)   #b, c, hw+HW

        value = self.value_fs_conv(torch.cat([feat0_flatten, feat1_flatten], dim=1))  # b, c, hw+HW
        value = (value.permute(0,2,1).contiguous().unsqueeze(2).repeat(1, 1, self.nhead, 1))   # b, hw+HW, nhead, c

        # print(value.shape)

        bs, _, h, w = query.shape
        query = query.flatten(2).permute(0,2,1).contiguous() # b, HW, c

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat0_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        

        for i, layer in enumerate(self.layers):
            query = layer(query, value,
                            height=h,
                            width=w,
                            query_location=query_location,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index,
                            )

        scale_feat = query.view(bs, height, width, self.d_model).permute(0,3,1,2).contiguous()

        return scale_feat


    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        #print(ref_2d.shape)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d


    
class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 num_head=1,
                 num_points = 8,
                 num_level=2,
                 dropout=0.1
                 ):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.num_points = num_points
        self.num_level = num_level
        self.num_head = num_head

        self.sampling_offsets = nn.Linear(d_model, num_head*num_level*num_points* 2)
        self.attention_weights = nn.Linear(d_model, num_head*num_level*num_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model*num_head, d_model)
        self.dropout = nn.Dropout(dropout)

        self.MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32

        self.init_weights()


    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_head,
            dtype=torch.float32) * (2.0 * math.pi / self.num_head)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_head, 1, 1,
            2).repeat(1, self.num_level, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self, query, value,
                height=None,
                width=None,
                query_location=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        '''
        query: [bs, hw, c]
        value: [bs, num_value, c]
        query_location: [bs, hw, num_level, 2]
        spatial_shapes: [2, 2]
        level_start_index: [2]
        '''

        bs, num_query, c = query.shape

        sampling_offsets = self.sampling_offsets(query)\
                .view(bs, num_query, self.num_head, self.num_level, self.num_points, 2)
        attention_weights = self.attention_weights(query)\
                .view(bs, num_query, self.num_head, self.num_level*self.num_points).softmax(-1)
        attention_weights = attention_weights\
                .view(bs, num_query, self.num_head, self.num_level, self.num_points)
    
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        # print(query_location[:, :, None, :, None, :].shape, sampling_offsets.shape, offset_normalizer[None, :, None, :].shape)
        sampling_locations = query_location[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, :, None, :]
        
        output = self.MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,attention_weights)
        #print(self.num_level, self.num_points)
        #print(self.num_head, output.shape)
        
        # output: (bs, num_query, c)
        output = self.output_proj(output)

        return self.dropout(output)

