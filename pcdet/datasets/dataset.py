from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .augmentor.test_augmentor import TestAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, is_source=True, root_path=None, logger=None,
                 da_train=False):
        super().__init__()
        self.num_frames=1
        self.test_flip = False
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.is_source = is_source
        self.da_train = da_train
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger,
            num_frames=self.num_frames
        ) if self.training else None

        test_aug_cfg = self.dataset_cfg.get('TEST_AUGMENTOR', None)

        if test_aug_cfg is not None:
            self.test_augmentor = TestAugmentor(test_aug_cfg, self.class_names, logger=self.logger,
                                            num_frames=self.num_frames)
        else:
            self.test_augmentor = None

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False


    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )
        else:
            if self.test_augmentor is not None:
                data_dict = self.test_augmentor.forward(
                    data_dict={
                        **data_dict,
                    }
                )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points','voxels_src', 'voxel_num_points_src']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'points_src', 'voxel_coords_src']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

class SeqDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.num_data_frames = dataset_cfg.NUM_FRAMES
        self.merge_frame = dataset_cfg.MERGE_FRAME

        if self.merge_frame:
            self.num_frames = 1
        else:
            self.num_frames = self.num_data_frames

        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger,num_frames=self.num_data_frames
        ) if self.training else None

        self.test_augmentor = TestAugmentor( self.dataset_cfg.TEST_AUGMENTOR, self.class_names,logger=self.logger,num_frames=self.num_data_frames)
        
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training,num_frames=self.num_data_frames
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.test_flip=False


    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    #@staticmethod
    def generate_prediction_dicts(self,batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def points_rigid_transform(self,cloud,pose):
        if cloud.shape[0]==0:
            return cloud
        mat=np.ones(shape=(cloud.shape[0],4),dtype=np.float32)
        pose_mat=np.mat(pose)
        mat[:,0:3]=cloud[:,0:3]
        mat=np.mat(mat)
        transformed_mat=pose_mat*mat.T
        T=np.array(transformed_mat.T,dtype=np.float32)
        return T[:,0:3]

    def get_registration_angle(self,mat):

        cos_theta=mat[0,0]
        sin_theta=mat[1,0]

        if  cos_theta < -1:
            cos_theta = -1
        if cos_theta > 1:
            cos_theta = 1

        theta_cos = np.arccos(cos_theta)

        if sin_theta >= 0:
            return theta_cos
        else:
            return 2 * np.pi - theta_cos

    def registration(self,data_dict):
        if 'gt_tracklets' in data_dict.keys():
            tracklets=data_dict['gt_tracklets']
        else:
            tracklets=None
        current_pose=data_dict['pose']
        inv_pose_of_last_frame = np.linalg.inv(current_pose)

        for i in range(self.num_data_frames-1):
            if 'points'+str(-i-1) in data_dict:
                this_points=data_dict['points'+str(-i-1)]
                this_pose = data_dict['pose' + str(-i - 1)]
                registration_mat = np.matmul(inv_pose_of_last_frame, this_pose)

                data_dict['points' + str(-i - 1)][:,0:3]=self.points_rigid_transform(this_points,registration_mat)[:,0:3]
                angle = self.get_registration_angle(registration_mat)

                if 'gt_tracklets' in data_dict.keys():

                    tracklets[:,7+i*4:10+i*4]=self.points_rigid_transform(tracklets[:,7+i*4:10+i*4],registration_mat)[:,0:3]
                    tracklets[:, 10 + i * 4]+=angle

                if 'gt_boxes'+str(-i-1) in data_dict:

                    data_dict['gt_boxes'+str(-i-1)][:,0:3]=self.points_rigid_transform(data_dict['gt_boxes'+str(-i-1)][:,:3],registration_mat)

                    data_dict['gt_boxes'+str(-i-1)][:,6] += angle

        if 'gt_tracklets' in data_dict.keys():
            data_dict['gt_tracklets']=tracklets

        return data_dict

    def prepare_multi_frame_data(self,data_dict):

        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """

        if self.training:
            if 'gt_boxes' not in data_dict:
                new_index = np.random.randint(self.__len__())
                data=self.__getitem__(new_index)
                return data

            data_dict = self.registration(data_dict)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
        else:
            data_dict = self.registration(data_dict)
            if self.test_augmentor is not None:
                data_dict = self.test_augmentor.forward(data_dict)


        if self.merge_frame:

            for i in range(self.num_data_frames):
                if i == 0:
                    points = data_dict['points']
                else:
                    points = data_dict['points' + str(-i)]
                new_points = np.zeros(shape=(points.shape[0], points.shape[1] + 1))
                new_points[:, 0:points.shape[1]] = points[:, :]
                new_points[:, -1] = self.num_data_frames-i

                if i == 0:
                    data_dict['points'] = new_points
                else:
                    data_dict['points' + str(-i)] = new_points


            all_points = []
            for i in range(self.num_data_frames):
                if i == 0 :
                    points = data_dict['points']
                else:
                    points = data_dict['points'+str(-i)]
                all_points.append(points)
            all_points = np.concatenate(all_points)
            data_dict['points'] = all_points


        if self.training:
            for i in range(self.num_data_frames):
                if i==0 and 'gt_boxes' in data_dict.keys():
                    gt_bbs = data_dict['gt_boxes']
                    ids = data_dict['ob_idx']

                    new_gt_bbs = np.zeros(shape=(gt_bbs.shape[0],gt_bbs.shape[1]+2),dtype=np.float32)
                    new_gt_bbs[:,0:7]=gt_bbs[:,0:7]
                    for j in range(len(gt_bbs)):
                        ob_id = ids[j]
                        if 'ob_idx'+str(-1) in data_dict.keys():
                            if ob_id in data_dict['ob_idx'+str(-1)]:
                                arg_id = data_dict['ob_idx'+str(-1)].index(ob_id)
                                new_gt_bbs[j,7:9] = (data_dict['gt_boxes'+str(-1)][arg_id,0:2]-gt_bbs[j,0:2])
                    data_dict['gt_boxes']=new_gt_bbs
                elif 'gt_boxes'+str(-i) in data_dict.keys():
                    gt_bbs = data_dict['gt_boxes'+str(-i)]
                    ids = data_dict['ob_idx'+str(-i)]
                    new_gt_bbs = np.zeros(shape=(gt_bbs.shape[0], gt_bbs.shape[1] + 2),dtype=np.float32)
                    new_gt_bbs[:, 0:7] = gt_bbs[:, 0:7]
                    for j in range(len(gt_bbs)):
                        ob_id = ids[j]
                        if 'ob_idx' + str(-i-1) in data_dict.keys():
                            if ob_id in data_dict['ob_idx' + str(-i-1)]:
                                arg_id = data_dict['ob_idx' + str(-i-1)].index(ob_id)
                                new_gt_bbs[j,7:9] = (data_dict['gt_boxes'+str(-i-1)][arg_id,0:2]-gt_bbs[j,0:2])
                    data_dict['gt_boxes'+str(-i)]=new_gt_bbs


        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            data_dict['gt_tracklets'] = data_dict['gt_tracklets'][selected]
            data_dict['num_bbs_in_tracklets'] = data_dict['num_bbs_in_tracklets'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes
            gt_tracklets = np.concatenate((data_dict['gt_tracklets'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_tracklets'] = gt_tracklets
            data_dict.pop('ob_idx', None)

        for i in range(1,self.num_data_frames):
            if data_dict.get('gt_boxes'+str(-i), None) is not None:
                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'+str(-i)], self.class_names)
                data_dict['gt_boxes'+str(-i)] = data_dict['gt_boxes'+str(-i)][selected]
                data_dict['gt_names'+str(-i)] = data_dict['gt_names'+str(-i)][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names'+str(-i)]], dtype=np.int32)

                gt_boxes = np.concatenate((data_dict['gt_boxes'+str(-i)], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes'+str(-i)] = gt_boxes
                data_dict.pop('gt_names'+str(-i), None)
                data_dict.pop('ob_idx'+str(-i), None)

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        data_dict.pop('gt_names', None)


        return data_dict

    def prepare_one_frame_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                }
            )
        else:
            if self.test_augmentor is not None:
                data_dict = self.test_augmentor.forward(
                    data_dict={
                        **data_dict,
                    }
                )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    def prepare_data(self, data_dict):
        if self.num_data_frames>1:
            return self.prepare_multi_frame_data(data_dict)
        else:
            return self.prepare_one_frame_data(data_dict)


    def collate_batch(self, batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        point_key_dict=['points', 'voxel_coords']
        for i in range(self.num_data_frames-1):
            point_key_dict.append('points'+str(-i-1))
            point_key_dict.append('voxel_coords'+str(-i-1))

        voxel_key_dict=['voxels', 'voxel_num_points']
        for i in range(self.num_data_frames-1):
            voxel_key_dict.append('voxels'+str(-i-1))
            voxel_key_dict.append('voxel_num_points' + str(-i - 1))

        gt_keys=['gt_boxes','gt_tracklets','num_bbs_in_tracklets']
        for i in range(1,self.num_data_frames):
            gt_keys.append('gt_boxes'+str(-i))

        for key, val in data_dict.items():
            try:
                if key in voxel_key_dict:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in point_key_dict:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in gt_keys:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(len(val)):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size

        return ret
