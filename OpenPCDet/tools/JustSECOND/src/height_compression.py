from typing import Dict

import torch
import torch.nn as nn

class HeightCompression(torch.jit.ScriptModule):
    def __init__(self, num_bev_features):
        super().__init__()
        self.num_bev_features = num_bev_features

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
