# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import tqdm
from pathlib import Path
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.datasets.dataset import SeqDatasetTemplate


class WaymoTrackingDataset(SeqDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.only_top_lidar = dataset_cfg.TL
        self.sampling = dataset_cfg.SAMPLING

        self.infos = []
        print(self.mode)
        self.include_waymo_data(self.mode)


    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        #self.infos = self.infos[0:500]
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        self.all_infos = self.infos

        self.index_list = list(range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in self.index_list:
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos

            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))


    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file[:-9]) + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        import concurrent.futures as futures
        from functools import partial
        # import sys
        # sys.path.append("./")
        import waymo_utils
        # from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        #with futures.ThreadPoolExecutor(num_workers) as executor:
        #    sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
        #                               total=len(sample_sequence_file_list)))
        all_sequences_infos=[]
        for i in tqdm.trange(len(sample_sequence_file_list)):
            single_sequence_file=sample_sequence_file_list[i]
            this_sequence_infos=process_single_sequence(single_sequence_file)
            all_sequences_infos+=this_sequence_infos
        #all_sequences_infos = [item for infos in sequence_infos for item in infos]

        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file).astype(np.float32)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        points_all = points_all[NLZ_flag == -1]
        points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def radius_sampling(self, points, dis=[5, 5, 5, 5], intev=[7, 5, 4, 2]):
        distance = np.sqrt(np.sum(points[:, 0:2] ** 2, 1))

        points_list = []

        dis_iter = 0

        for i in range(len(dis)):
            dis_thresh = dis[i]
            sample_interval = intev[i]

            pos1 = dis_iter < distance
            pos2 = distance <= dis_iter + dis_thresh

            this_points = points[pos1 * pos2]
            sampling_flag = np.arange(0, this_points.shape[0])
            sampling_flag = sampling_flag % sample_interval == 0
            sampling_flag = sampling_flag.astype(np.bool)
            this_points = this_points[sampling_flag]
            points_list.append(this_points)
            dis_iter += dis_thresh

        points_list.append(points[distance > dis_iter])

        return np.concatenate(points_list)

    def get_multi_frame_data(self,index):

        num_frames=self.num_data_frames

        assert num_frames>=2, "at least two frames required"

        all_points={}
        all_poses={}
        points=None
        pose=None

        current_frame_id=self.all_infos[index]['point_cloud']['sample_idx']
        current_seq_id=self.all_infos[index]['point_cloud']['lidar_sequence']

        all_gt_boxes_lidar={}
        all_gt_ob_idx={}
        all_gt_names={}
        current_gt_names=[]
        current_gt_boxes=[]
        current_gt_ob_idx=[]

        past_gt_boxes={}
        past_gt_names={}
        past_gt_ob_idx={}

        valid_idx = index
        for i in range(0,num_frames):
            global_idx = index-i
            if global_idx<0:
                global_idx = index

            info = copy.deepcopy(self.all_infos[global_idx])

            sample_idx = info['point_cloud']['sample_idx']
            seq_idx = info['point_cloud']['lidar_sequence']
            num_points_of_each_lidar = info['num_points_of_each_lidar']

            if seq_idx==current_seq_id:
                valid_idx = global_idx
            else:
                info = copy.deepcopy(self.all_infos[valid_idx])
                sample_idx = info['point_cloud']['sample_idx']
                seq_idx = info['point_cloud']['lidar_sequence']
                #num_points_of_each_lidar = info['num_points_of_each_lidar']

            this_points = self.get_lidar(seq_idx, sample_idx)
            #if self.only_top_lidar:
            #    this_points = this_points[0:num_points_of_each_lidar[0]]
            #if self.sampling:
            #    this_points = self.radius_sampling(this_points)

            this_pose = info['pose']

            all_points[-i]=this_points
            all_poses[-i]=this_pose

            if i==0:
                points=this_points
                pose=this_pose

            if 'annos' in info:
                annos = info['annos']
                if annos is None:
                    continue
                annos = common_utils.drop_info_with_name(annos, name='unknown')
                gt_names = annos['name']
                gt_boxes_lidar = annos['gt_boxes_lidar']
                ob_idx = annos['obj_ids']

                num_points_in_gt = annos.get('num_points_in_gt', None)

                if num_points_in_gt is not None:
                    mask = num_points_in_gt[:] > 0
                    gt_names = gt_names[mask]
                    gt_boxes_lidar = gt_boxes_lidar[mask]
                    ob_idx = ob_idx[mask]

                all_gt_boxes_lidar[-i]=gt_boxes_lidar
                all_gt_ob_idx[-i] = ob_idx
                all_gt_names[-i] = gt_names
                if i==0:
                    current_gt_boxes=gt_boxes_lidar
                    current_gt_names=gt_names
                    current_gt_ob_idx=ob_idx.tolist()
                else:
                    past_gt_boxes[-i]=gt_boxes_lidar
                    past_gt_names[-i]=gt_names
                    past_gt_ob_idx[-i]=ob_idx.tolist()

        gt_tracklets = []
        num_bbs_in_tracklets = []

        for i, ob_id in enumerate(current_gt_ob_idx):

            tracklet = np.zeros(shape=(7 + (num_frames - 1) * 4))
            tracklet[0:7] = current_gt_boxes[i][0:7]

            nums = 1

            for j in range(num_frames - 1):
                frame_id = -j - 1
                t_id = 7 + j * 4
                if frame_id in all_gt_ob_idx.keys():
                    this_gt_ob_idx = all_gt_ob_idx[frame_id].tolist()
                    if ob_id in this_gt_ob_idx:
                        arg_id = this_gt_ob_idx.index(ob_id)
                        this_box = all_gt_boxes_lidar[frame_id][arg_id]
                        tracklet[t_id:t_id + 3] = this_box[0:3]
                        tracklet[t_id + 3] = this_box[6]
                        nums += 1
            num_bbs_in_tracklets.append([nums])
            gt_tracklets.append(tracklet)

        if len(gt_tracklets) == 0:
            gt_tracklets = np.zeros(shape=(0, 7 + (num_frames - 1) * 4))
            num_bbs_in_tracklets = np.zeros(shape=(0, 1))
            current_gt_names = np.zeros(shape=(0,))
            current_gt_boxes = np.zeros(shape=(0, 7))
            current_gt_ob_idx = []
        else:
            gt_tracklets = np.array(gt_tracklets)
            num_bbs_in_tracklets = np.array(num_bbs_in_tracklets)
            current_gt_names = np.array(current_gt_names)
            current_gt_boxes = np.array(current_gt_boxes)

        for i in range(1, num_frames):
            if -i not in past_gt_boxes.keys():
                past_gt_boxes[-i] = np.zeros(shape=(0, 7))
                past_gt_names[-i] = np.zeros(shape=(0,))
                past_gt_ob_idx[-i] = []
            else:
                past_gt_boxes[-i] = np.array(past_gt_boxes[-i])
                past_gt_names[-i] = np.array(past_gt_names[-i])


        input_dict={
            'points': points,
            'seq_id': current_seq_id,
            'frame_id': current_frame_id,
            'pose': pose
        }

        for i in range(num_frames-1):
            if -i-1 in all_points:
                input_dict.update({"points"+str(-i-1): all_points[-i-1]})
            else:
                input_dict.update({"points" + str(-i - 1): np.zeros(shape=(0,5))})
            if -i-1 in all_poses:
                input_dict.update({"pose"+str(-i-1): all_poses[-i-1]})
            else:
                input_dict.update({"pose" + str(-i - 1): np.array(np.eye(5))})



        input_dict.update({
            'gt_names': current_gt_names,
            'gt_boxes': current_gt_boxes,
            'gt_tracklets': gt_tracklets,
            'num_bbs_in_tracklets': num_bbs_in_tracklets,
            'ob_idx': current_gt_ob_idx
        })

        for i in range(1, num_frames):

            input_dict.update({"gt_boxes"+str(-i):past_gt_boxes[-i]})
            input_dict.update({"gt_names" + str(-i): past_gt_names[-i]})
            input_dict.update({'ob_idx' + str(-i): past_gt_ob_idx[-i]})

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = self.all_infos[index].get('metadata', self.all_infos[index]['frame_id'])

        return data_dict

    def get_one_frame_data(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.all_infos)

        info = copy.deepcopy(self.all_infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = self.get_lidar(sequence_name, sample_idx)

        current_frame_id=self.all_infos[index]['point_cloud']['sample_idx']
        current_seq_id=self.all_infos[index]['point_cloud']['lidar_sequence']


        input_dict = {
            'points': points,
            'frame_id': current_frame_id,
            'seq_id': current_seq_id
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            num_points_in_gt = annos.get('num_points_in_gt', None)

            if num_points_in_gt is not None:
                mask = num_points_in_gt[:]>0
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                num_points_in_gt = num_points_in_gt[mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': num_points_in_gt
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)

        return data_dict
    def __getitem__(self, index):

        index = self.index_list[index]

        if self.num_data_frames == 1:
            data_dict = self.get_one_frame_data(index)
        else:
            data_dict=self.get_multi_frame_data(index)
        return data_dict

    #@staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)

            if self.test_augmentor is not None:
                single_pred_dict = self.test_augmentor.backward(single_pred_dict)

            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['seq_id'] = batch_dict['seq_id'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)

        eval_gt_annos = []

        for i in range(len(self.infos)):
            info = copy.deepcopy(self.infos[i])
            gt_annos = info['annos']
            frame_id = info['point_cloud']['sample_idx']
            seq_id = info['point_cloud']['lidar_sequence']
            gt_annos['frame_id'] = frame_id
            gt_annos['seq_id'] = seq_id
            eval_gt_annos.append(gt_annos)

        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_track_groundtruth_database(self, info_path, save_path, used_classes=None, split='train'):
        gt_path_name = Path('pcdet_gt_track_database_%s_cp' % split)

        database_save_path = save_path / gt_path_name


        db_info_save_path = save_path / ('pcdet_waymo_track_dbinfos_%s_cp.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(0, len(infos)):

            print('tracks_gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']

            if names.shape[0] == 0:
                continue

            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']
            obj_ids = annos['obj_ids']

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            this_file = database_save_path / sequence_name / str(sample_idx)

            os.makedirs(this_file, exist_ok=True)

            for i in range(num_obj):
                ob_id = obj_ids[i]
                filename = '%s_%s.bin' % (names[i], ob_id)
                filepath = database_save_path / sequence_name / str(sample_idx) / filename

                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:

                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

            if k % 16 == 0:
                mask_car = (names == 'Vehicle')
            else:
                mask_car = np.zeros(shape=names.shape).astype(np.bool)

            if k % 10 == 0:
                mask_p = (names == 'Pedestrian')
            else:
                mask_p = np.zeros(shape=names.shape).astype(np.bool)

            if k % 2 == 0:
                mask_c = (names == 'Cyclist')
            else:
                mask_c = np.zeros(shape=names.shape).astype(np.bool)


            mask_c = (names == 'Cyclist')

            mask = mask_car+mask_p+mask_c

            names = names[mask]
            gt_boxes = gt_boxes[mask]
            difficulty = difficulty[mask]
            obj_ids = obj_ids[mask]

            num_obj = gt_boxes.shape[0]
            if num_obj ==0:
                continue

            for i in range(num_obj):

                ob_id = obj_ids[i]
                filename = '%s_%s.bin' % (names[i], ob_id)
                filepath = database_save_path / sequence_name / str(sample_idx) / filename
                gt_points = np.fromfile(str(filepath), dtype=np.float32).reshape(
                    [-1, 5])

                if gt_points.shape[0]<=0:
                    continue

                if (used_classes is None) or names[i] in used_classes:

                    db_path = str(gt_path_name / sequence_name / str(sample_idx) / filename)  # gt_database/xxxxx.bin

                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               "seq_idx": sequence_name, 'image_idx': sample_idx,'sample_idx': sample_idx,
                               'gt_idx': i, 'ob_idx': ob_id,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0], 'pose': info['pose'],
                               'difficulty': difficulty[i]}

                    for pre_i in range(5):
                        pre_k=k-pre_i-1
                        if pre_k<0:
                            continue
                        pre_info = infos[pre_k]
                        pre_pc_info = pre_info['point_cloud']
                        pre_sequence_name = pre_pc_info['lidar_sequence']
                        pre_sample_idx = pre_pc_info['sample_idx']
                        pre_pose = pre_info['pose']

                        if pre_sequence_name != sequence_name:
                            continue

                        pre_annos = pre_info['annos']
                        if pre_annos is None:
                            continue

                        pre_gt_boxes = pre_annos['gt_boxes_lidar']
                        pre_ob_idx = pre_annos['obj_ids']
                        pre_ob_idx = pre_ob_idx.tolist()

                        if ob_id in pre_ob_idx:
                            pre_ob_key = pre_ob_idx.index(ob_id)

                            pre_filepath = str(gt_path_name / sequence_name / str(pre_sample_idx) / filename)

                            thisfilepath = database_save_path / sequence_name / str(pre_sample_idx) / filename

                            gt_points = np.fromfile(str(thisfilepath), dtype=np.float32).reshape(
                                [-1, 5])

                            if gt_points.shape[0] <= 0:
                                continue

                            db_info.update({'box3d_lidar' + str(-pre_i - 1): pre_gt_boxes[pre_ob_key]})
                            db_info.update({'path' + str(-pre_i - 1): pre_filepath})
                            db_info.update({'pose' + str(-pre_i - 1): pre_pose})
                        else:
                            continue

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]


        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count()):
    # multiprocessing.cpu_count()
    # https://blog.csdn.net/qq_30159015/article/details/82658896

    dataset = WaymoTrackingDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)
    test_filename = save_path / ('waymo_infos_%s.pkl' % test_split)

    print('---------------Start to generate data infos---------------')
   
    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)


    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)


    dataset.set_split(test_split)
    waymo_infos_test = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=False,
        sampled_interval=1
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(waymo_infos_test, f)
    print('----------------Waymo info val file is saved to %s----------------' % test_filename)



    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)

    dataset.create_track_groundtruth_database(
        info_path=train_filename, save_path=save_path, split='train',
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist']
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='waymo_tracking_dataset.yaml', help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )

