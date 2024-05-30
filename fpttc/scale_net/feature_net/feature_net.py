import torch
import torch.nn as nn
import torch.nn.functional as F


from .feature_trans import FeatureTransformer, FeatureTransformerS
from ...modules.utils import normalize_img, feature_add_position, upsample_flow_with_mask
from ...modules.geometry import flow_wrap
from ...modules.matching import (global_correlation_softmax, local_scale_correlation, local_correlation_with_flow)


class FeatureNet(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 ):
        super(FeatureNet, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

    def forward(self, feature0, feature1, 
                scale_idx, 
                attn_type=None,
                attn_splits_list=None,
                flow=None,
                flow_back=None,
                testing=False
                ):

        attn_splits = attn_splits_list[scale_idx]
        if not testing or scale_idx<2:
            if flow is not None:
                flow = flow.detach()
                feature1_f = flow_wrap(feature1, flow)

            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                    attn_type=attn_type,
                                                    attn_num_splits=attn_splits,
                                                    )
        else:
            flow = flow.detach()
            feature1_f = flow_wrap(feature1, flow)
            flow_back = flow_back.detach()
            feature0_b = flow_wrap(feature0, flow_back)

            feature0, feature1 = torch.cat((feature0, feature1_f), dim=0), torch.cat((feature1, feature0_b), dim=0)

            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1,
                                                    attn_type=attn_type,
                                                    attn_num_splits=attn_splits,
                                                    )            

        return feature0, feature1



class GmaAtten(nn.Module):
    def __init__(self,
                 num_scales=2,
                 feature_channels=128,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 ):
        super(GmaAtten, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales

        # Transformer
        self.transformer = FeatureTransformerS(num_layers=1,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )
    def forward(self, feature0, feature1, 
                attn_type=None,
                attn_splits_list=None,
                flow=None,
                flow_back=None,
                testing=False
                ):

        attn_splits = attn_splits_list[-1]

        feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

        # Transformer
        agg_feat = self.transformer(feature0, feature1,
                                    attn_type=attn_type,
                                    attn_num_splits=attn_splits,
                                    )

        return agg_feat

    # def forward(self, feature0, feature1, 
    #             scale_idx, 
    #             attn_type=None,
    #             attn_splits_list=None,
    #             flow=None,
    #             flow_back=None,
    #             testing=False
    #             ):

    #     attn_splits = attn_splits_list[scale_idx]
    #     if not testing or scale_idx<2:
    #         if flow is not None:
    #             flow = flow.detach()
    #             feature1_f = flow_wrap(feature1, flow)

    #         feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

    #         # Transformer
    #         feature0, feature1 = self.transformer(feature0, feature1,
    #                                                 attn_type=attn_type,
    #                                                 attn_num_splits=attn_splits,
    #                                                 )
    #     else:
    #         flow = flow.detach()
    #         feature1_f = flow_wrap(feature1, flow)
    #         flow_back = flow_back.detach()
    #         feature0_b = flow_wrap(feature0, flow_back)

    #         feature0, feature1 = torch.cat((feature0, feature1_f), dim=0), torch.cat((feature1, feature0_b), dim=0)

    #         feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

    #         # Transformer
    #         feature0, feature1 = self.transformer(feature0, feature1,
    #                                                 attn_type=attn_type,
    #                                                 attn_num_splits=attn_splits,
    #                                                 )            

    #     return feature0, feature1

