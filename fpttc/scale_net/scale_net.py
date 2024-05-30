import torch
import torch.nn as nn
import torch.nn.functional as F

from .scale_encoder import ScaleEncoder
from .decoder import ScaleDecoder
from ..modules.attention import SelfAttnPropagation
from ..modules.geometry import flow_wrap

from .feature_net.feature_net import GmaAtten

class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super(FeatureFusion, self).__init__()
        self.convc1 = nn.Conv2d(dim, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(dim, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, dim, 3, padding=1)
    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return out


class ScaleNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 upsample_factor=4,
                 num_head=1,
                 num_transformer_layers=6,
                 scale_level=1,
                 reg_refine=False,  # optional local regression refinement
                 query_lvl = -1
                 ):
        super(ScaleNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # the level of features, flows while generating query
        self.query_lvl = query_lvl

        # Transformer
        self.encoder = ScaleEncoder(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              num_feature_levels=num_scales,
                                              num_level=scale_level
                                              )

        # propagation with self-attn
        self.decoder = ScaleDecoder(upsample_factor=upsample_factor)

        # self.atten = GMA(d_model=feature_channels)
        self.gma = GmaAtten(num_scales=1, feature_channels=feature_channels, 
                                  num_head=1, ffn_dim_expansion=2, 
                                  num_transformer_layers=1)

    def forward(self, corr, 
                    feature0_listc, feature1_listc, ini_scale=None
                ):

        cfeat0 = feature0_listc[-1]

        scale_feature = self.encoder(feature0_listc, feature1_listc, \
                                    self.query_lvl, ini_query=corr)     #(bs, c, h, w)
        
        agg_corr = self.gma(scale_feature, corr, 'swin', [2,8])

        scale = self.decoder(scale_feature, cfeat0, agg_corr, ini_scale)

        return scale



