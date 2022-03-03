import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg,num_frames, **kwargs):
        super().__init__()
        self.num_frames=num_frames
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
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
        if self.num_frames>1:
            for i in range(self.num_frames-1):
                if 'encoded_spconv_tensor'+str(-i-1) in batch_dict:
                    encoded_spconv_tensor = batch_dict['encoded_spconv_tensor'+str(-i-1)]
                    spatial_features = encoded_spconv_tensor.dense()
                    N, C, D, H, W = spatial_features.shape
                    spatial_features = spatial_features.view(N, C * D, H, W)
                    batch_dict['spatial_features'+str(-i-1)] = spatial_features


        return batch_dict

