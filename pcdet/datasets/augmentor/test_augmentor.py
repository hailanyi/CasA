from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils


class TestAugmentor(object):
    def __init__(self, augmentor_configs, class_names, logger=None, num_frames=1):
        self.class_names = class_names
        self.logger = logger
        self.num_frames=num_frames

        self.data_augmentor_queue = []
        self.test_back_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for i,cur_cfg in enumerate(aug_config_list):
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
            back_config = aug_config_list[-(i+1)]
            cur_augmentor = getattr(self, back_config.NAME)(config=back_config)
            self.test_back_queue.append(cur_augmentor)


    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def world_flip(self, data_dict=None, config=None):

        if data_dict is None:
            return partial(self.world_flip, config=config)

        axis = config['ALONG_AXIS']

        if axis is None:
            return data_dict

        for i in range(self.num_frames):
            if i == 0:
                frame_id = ''
            else:
                frame_id = str(-i)

            if 'points'+frame_id in data_dict:
                points = data_dict['points'+frame_id]
                if axis == 'x':
                    points = getattr(augmentor_utils, 'random_flip_with_param')(
                        points, True, ax=1)
                if axis == 'y':
                    points = getattr(augmentor_utils, 'random_flip_with_param')(
                        points, True, ax=0)
                data_dict['points'+frame_id]=points

            if 'boxes_lidar'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_lidar'+frame_id]
                if axis == 'x':
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=1)
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=6)
                if axis == 'y':
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=0)
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=6, offset=np.pi)
                data_dict['boxes_lidar'+frame_id] = boxes_lidar
            if 'boxes_3d'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_3d'+frame_id]
                if axis == 'x':
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=1)
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=6)
                if axis == 'y':
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=0)
                    boxes_lidar = getattr(augmentor_utils, 'random_flip_with_param')(
                        boxes_lidar, True, ax=6, offset=np.pi)
                data_dict['boxes_3d'+frame_id] = boxes_lidar


        return data_dict

    def world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_rotation, config=config)
        rot_factor = config['WORLD_ROT']

        for i in range(self.num_frames):
            if i == 0:
                frame_id = ''
            else:
                frame_id = str(-i)

            if 'points'+frame_id in data_dict:
                points = data_dict['points'+frame_id]
                points[:,0:3] = common_utils.rotate_points_along_z(points[np.newaxis, :, 0:3], np.array([rot_factor]))[
                    0]
                data_dict['points' + frame_id] = points

            if 'boxes_lidar'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_lidar'+frame_id]
                boxes_lidar[:,0:3] = common_utils.rotate_points_along_z(boxes_lidar[np.newaxis, :, 0:3], np.array([-rot_factor]))[
                    0]
                boxes_lidar[:,6] += -rot_factor
                data_dict['boxes_lidar'+frame_id] = boxes_lidar

            if 'boxes_3d'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_3d'+frame_id]
                boxes_lidar[:,0:3] = common_utils.rotate_points_along_z(boxes_lidar[np.newaxis, :, 0:3], np.array([-rot_factor]))[
                    0]
                boxes_lidar[:,6] += -rot_factor
                data_dict['boxes_3d'+frame_id] = boxes_lidar

        return data_dict

    def world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_scaling, config=config)
        scale_factor = config['WORLD_SCALE']

        for i in range(self.num_frames):
            if i == 0:
                frame_id = ''
            else:
                frame_id = str(-i)

            if 'points'+frame_id in data_dict:
                points = data_dict['points'+frame_id]
                points[:,0:3]*=scale_factor
                data_dict['points' + frame_id] = points

            if 'boxes_lidar'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_lidar'+frame_id]
                boxes_lidar[:,0:6]/=scale_factor
                data_dict['boxes_lidar' + frame_id] = boxes_lidar
            if 'boxes_3d'+frame_id in data_dict:
                boxes_lidar = data_dict['boxes_3d'+frame_id]
                boxes_lidar[:,0:6]/=scale_factor
                data_dict['boxes_3d' + frame_id] = boxes_lidar
        return  data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict

    def backward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.test_back_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        return data_dict
