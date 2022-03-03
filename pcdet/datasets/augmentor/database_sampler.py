import pathlib
import pickle

import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
import time

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names,num_frames, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        #self.gt_path = pathlib.Path(sampler_cfg.GT_PATH)

        self.logger = logger
        self.db_infos = {}
        self.num_frames=num_frames
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:

            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                for cls in class_names:
                    if cls in infos.keys():
                        self.db_infos[cls].extend(infos[cls])
                #[self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            this_infos = []
            for info in dinfos:
                if 'difficulty' in info:
                    if info['difficulty'] not in removed_difficulty:
                        this_infos.append(info)
                else:
                    this_infos.append(info)
            new_db_infos[key] = this_infos
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

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

    def registration(self,pose, pre_pose, pre_obj_points, pre_box3d_lidar):

        inv_pose_of_last_frame = np.linalg.inv(pose)
        registration_mat = np.matmul(inv_pose_of_last_frame, pre_pose)

        if len(pre_obj_points)!=0:
            pre_obj_points[:, 0:3] = self.points_rigid_transform(pre_obj_points, registration_mat)[:,0:3]
        angle = self.get_registration_angle(registration_mat)
        pre_box3d_lidar[0:3] = self.points_rigid_transform(np.array([pre_box3d_lidar]), registration_mat)[0, 0:3]
        pre_box3d_lidar[6]+=angle

        return pre_obj_points, pre_box3d_lidar

    def add_sampled_boxes_to_scene_multi(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        gt_idx = np.array(data_dict['ob_idx'])[gt_boxes_mask]

        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets']=data_dict['gt_tracklets'][gt_boxes_mask]
            data_dict['num_bbs_in_tracklets'] = data_dict['num_bbs_in_tracklets'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        ob_index_list =[]
        box3d_lidar_list = []

        for idx, info in enumerate(total_valid_sampled_dict):

            path = pathlib.Path(self.root_path)
            file_path = path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            pre_box3d_lidar = info['box3d_lidar']
            id = info['ob_idx']
            seq_idx = info["seq_idx"]
            sample_idx = info['image_idx']

            obj_points_list.append(obj_points)
            ob_index_list.append(str(sample_idx)+'_'+str(seq_idx)+'_'+str(id))
            box3d_lidar_list.append(pre_box3d_lidar)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        ob_idx = np.array(ob_index_list)
        sampled_gt_boxes = np.array(box3d_lidar_list)

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        gt_idx = np.concatenate([gt_idx, ob_idx], axis=0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['ob_idx'] = gt_idx.tolist()

        if self.num_frames>1:

            gt_tracklets = np.zeros(shape=(len(total_valid_sampled_dict),7+(self.num_frames-1)*4))

            gt_tracklets[:,0:7]=sampled_gt_boxes[:,0:7]

            num_bbs_in_tracks = np.ones(shape=(len(total_valid_sampled_dict),1))

            for i in range(1,self.num_frames):

                if 'points'+str(-i) not in data_dict:
                    continue
                if 'gt_names'+str(-i) not in data_dict:
                    pre_gt_boxes = np.zeros(shape=(0,7))
                    pre_gt_names = np.zeros(shape=(0,))
                    pre_gt_idx = np.zeros(shape=(0,))
                else:
                    pre_gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names'+str(-i)]], dtype=np.bool_)
                    pre_gt_boxes = data_dict['gt_boxes'+str(-i)][pre_gt_boxes_mask]
                    pre_gt_names = data_dict['gt_names'+str(-i)][pre_gt_boxes_mask]
                    pre_gt_idx = np.array(data_dict['ob_idx' + str(-i)])[pre_gt_boxes_mask]
                pre_points = data_dict['points'+str(-i)]

                pre_obj_points_list = []
                pre_box3d_lidar_list=[]
                pre_sampled_gt_names=[]
                pre_ob_idx_list=[]

                for idx, info in enumerate(total_valid_sampled_dict):

                    if 'box3d_lidar'+str(-i) in info:
                        num_bbs_in_tracks[idx,0]+=1
                        pre_box3d_lidar = np.zeros(shape=info['box3d_lidar'+str(-i)].shape)
                        pre_box3d_lidar[:] = info['box3d_lidar'+str(-i)][:]

                        #path_str = str(info['path'+str(-i)])
                        #path_str = path_str.split('/')

                        #pre_file_path = self.root_path / path_str[-2] / path_str[-1]
                        this_path = pathlib.Path(self.root_path)
                        pre_file_path = this_path/info['path'+str(-i)]

                        pre_obj_points = np.fromfile(str(pre_file_path), dtype=np.float32).reshape(
                            [-1, self.sampler_cfg.NUM_POINT_FEATURES])

                        pre_obj_points[:, :3] += pre_box3d_lidar[:3]

                        pose = info['pose']
                        pre_pose = info['pose'+str(-i)]

                        pre_obj_points,pre_box3d_lidar = self.registration(pose,pre_pose,pre_obj_points,pre_box3d_lidar)

                        gt_tracklets[idx,3+i*4:6+i*4]=pre_box3d_lidar[0:3]
                        gt_tracklets[idx, 10+(i-1)*4] = pre_box3d_lidar[6]

                        pre_box3d_lidar_list.append(pre_box3d_lidar)
                        pre_obj_points_list.append(pre_obj_points)
                        pre_sampled_gt_names.append(info['name'])

                        id = info['ob_idx']
                        seq_idx = info["seq_idx"]
                        sample_idx = info['image_idx']
                        pre_ob_idx_list.append(str(sample_idx)+'_'+str(seq_idx)+'_'+str(id))

                if len(pre_obj_points_list)>0:
                    pre_obj_points = np.concatenate(pre_obj_points_list, axis=0)
                    pre_box3d_lidar = np.array(pre_box3d_lidar_list)
                    pre_ob_idx = np.array(pre_ob_idx_list)
                    pre_sampled_gt_names=np.array(pre_sampled_gt_names)

                    pre_points = box_utils.remove_points_in_boxes3d(pre_points, pre_box3d_lidar)
                    pre_points = np.concatenate([pre_points, pre_obj_points], axis=0)
                    pre_gt_names = np.concatenate([pre_gt_names, pre_sampled_gt_names], axis=0)
                    pre_gt_boxes = np.concatenate([pre_gt_boxes, pre_box3d_lidar], axis=0)
                    pre_gt_idx = np.concatenate([pre_gt_idx,pre_ob_idx],0)

                    data_dict['gt_boxes'+str(-i)] = pre_gt_boxes
                    data_dict['gt_names'+str(-i)] = pre_gt_names
                    data_dict['points'+str(-i)] = pre_points
                    data_dict['ob_idx'+str(-i)] = pre_gt_idx.tolist()

            if 'gt_tracklets' in data_dict:
                data_dict["gt_tracklets"]=np.concatenate([data_dict["gt_tracklets"],gt_tracklets], axis=0)
                data_dict["num_bbs_in_tracklets"] = np.concatenate([data_dict["num_bbs_in_tracklets"], num_bbs_in_tracks], axis=0)

        return data_dict

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):

        gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        if 'gt_tracklets' in data_dict:
            data_dict['gt_tracklets']=data_dict['gt_tracklets'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        for idx, info in enumerate(total_valid_sampled_dict):

            file_path = self.root_path / info['path']
            #path = pathlib.Path(self.root_path)
            #file_path = path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """

        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])

                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            if self.num_frames>1:
                data_dict = self.add_sampled_boxes_to_scene_multi(data_dict, sampled_gt_boxes, total_valid_sampled_dict)
            else:
                data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)


        return data_dict
