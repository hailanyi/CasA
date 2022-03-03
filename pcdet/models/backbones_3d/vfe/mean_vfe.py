import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, num_frames, **kwargs):
        super().__init__(model_cfg=model_cfg,num_frames=num_frames)
        self.num_point_features = num_point_features
        self.model = self.model_cfg.get('MODEL',None)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        if 'semi_test' in batch_dict:
            voxel_features, voxel_num_points = batch_dict['voxels_src'], batch_dict[
                'voxel_num_points_src' ]
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer

            if self.model is not None:
                if self.model == 'max':
                    time_max = voxel_features[:, :, :].max(dim=1, keepdim=False)[0]
                    points_mean[:, -1] = time_max[:, -1]

            batch_dict['voxel_features_src'] = points_mean.contiguous()

        for i in range(self.num_frames):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(-i)

            voxel_features, voxel_num_points = batch_dict['voxels'+frame_id], batch_dict['voxel_num_points'+frame_id]
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer

            if self.model is not None:
                if self.model == 'max':
                    time_max = voxel_features[:, :, :].max(dim=1, keepdim=False)[0]
                    points_mean[:,-1]=time_max[:,-1]

            batch_dict['voxel_features'+frame_id] = points_mean.contiguous()

        return batch_dict
