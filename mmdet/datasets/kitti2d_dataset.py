# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import copy

from mmdet.datasets import DATASETS, CustomDataset
from mmcv.utils import print_log


@DATASETS.register_module()
class Kitti2DDataset(CustomDataset):
    r"""KITTI 2D Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        evaluation_ids (List[int], optional): Specify class indices to use 
            for evaluation. If is None, all classes will be used. Default: None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        lidar_prefix= (str, optional) prefix for liadar images,
        lidar_img_mode= (bool, optional) whether to use lidar image processing. 
            Default: False,
        radar_prefix= (str, optional) radar for liadar images,
        radar_img_mode= (bool, optional) whether to use radar image processing. 
            Default: False,
    """

    CLASSES = ('car', 'pedestrian', 'cyclist')
    """
    Annotation format:
    [
        {
            'image': {
                'image_idx': 0,
                'image_path': 'training/image_2/000000.png',
                'image_shape': array([ 370, 1224], dtype=int32)
            },
            'point_cloud': {
                 'num_features': 4,
                 'velodyne_path': 'training/velodyne/000000.bin'
             },
             'calib': {
                 'P0': <np.ndarray> (4, 4),
                 'P1': <np.ndarray> (4, 4),
                 'P2': <np.ndarray> (4, 4),
                 'P3': <np.ndarray> (4, 4),
                 'R0_rect':4x4 np.array,
                 'Tr_velo_to_cam': 4x4 np.array,
                 'Tr_imu_to_velo': 4x4 np.array
             },
             'annos': {
                 'name': <np.ndarray> (n),
                 'truncated': <np.ndarray> (n),
                 'occluded': <np.ndarray> (n),
                 'alpha': <np.ndarray> (n),
                 'bbox': <np.ndarray> (n, 4),
                 'dimensions': <np.ndarray> (n, 3),
                 'location': <np.ndarray> (n, 3),
                 'rotation_y': <np.ndarray> (n),
                 'score': <np.ndarray> (n),
                 'index': array([0], dtype=int32),
                 'group_ids': array([0], dtype=int32),
                 'difficulty': array([0], dtype=int32),
                 'num_points_in_gt': <np.ndarray> (n),
             }
        }
    ]
    """

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        self.data_infos = mmcv.load(ann_file)
        self.cat2label = {
            cat_name: i
            for i, cat_name in enumerate(self.CLASSES)
        }
        return self.data_infos
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['image']['image_shape'][1] / img_info['image']['image_shape'][0] > 1:
                self.flag[i] = 1

    def _filter_imgs(self, min_size=32):
        """Filter images without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if len(img_info['annos']['name']) == 1 and img_info['annos']['name'][0] == 'ignore':
                continue
            if len(img_info['annos']['name']) > 0:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - bboxes (np.ndarray): Ground truth bboxes.
                - labels (np.ndarray): Labels of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        annos = info['annos']
        gt_names = annos['name']
        gt_bboxes = annos['bbox']
        difficulty = annos['difficulty']

        # remove classes that is not needed
        selected = self.keep_arrays_by_name(gt_names, self.CLASSES)
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_labels = np.array([self.cat2label[n] for n in gt_names])

        anns_results = dict(
            bboxes=gt_bboxes.astype(np.float32),
            labels=gt_labels,
        )
        return anns_results

    def prepare_train_img(self, idx):
        """Training image preparation.

        Args:
            index (int): Index for accessing the target image data.

        Returns:
            dict: Training image data dict after preprocessing
                corresponding to the index.
        """
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        ann_info = self.get_ann_info(idx)
        if len(ann_info['bboxes']) == 0:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]        
        if self.lidar_img_mode:
            results['lidar_info'] = self.get_sensor_info(idx, sensor_projection='lidar_projections')
        if self.radar_img_mode:
            results['radar_info'] = self.get_sensor_info(idx, sensor_projection='radar_projections') 
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target image data.

        Returns:
            dict: Testing image data dict after preprocessing
                corresponding to the index.
        """
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        if self.lidar_img_mode:
            results['lidar_info'] = self.get_sensor_info(idx, sensor_projection='lidar_projections')
        if self.radar_img_mode:
            results['radar_info'] = self.get_sensor_info(idx, sensor_projection='radar_projections') 
        self.pre_pipeline(results)
        return self.pipeline(results)

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def reformat_bbox(self, outputs, out=None):
        """Reformat bounding boxes to KITTI 2D styles.

        Args:
            outputs (list[np.ndarray]): List of arrays storing the inferenced
                bounding boxes and scores.
            out (str | None): The prefix of output file. Default: None.

        Returns:
            list[dict]: A list of dictionaries with the kitti 2D format.
        """
        # not for STF: sample_idx = [info['image']['image_idx'] for info in self.data_infos]
        result_files = self.bbox2result_kitti2d(outputs, self.CLASSES, None,
                                           out)
        return result_files
    

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the \
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.data_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = sample_idx # orig but not needed: np.array([sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def evaluate(self, result_files, logger=None, metric=None, pklfile_prefix=None, eval_on_crop=False):
        """Evaluation in KITTI protocol.

        Args:
            result_files (str): Path of result files.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Types of evaluation. Default: None.
                KITTI dataset only support 'bbox' evaluation type.
            pklfile_prefix (str | None): The prefix of pkl file.
            eval_on_crop (dict | False): Dictionary containing 
                parameters for evaluating on a specific crop including:
                'offset_h', 'offset_w', 'img_shape', 'thresh_in_frame'

        Returns:
            tuple (str, dict): Average precision results in str format
                and average precision results in dict format.
        """
        result_files = self.reformat_bbox(result_files)
        from mmdet.core.evaluation import kitti_eval
        metric = ['bbox'] if not metric else metric
        assert metric in ('bbox', ['bbox']), 'KITTI 2D data set only evaluate bbox'
        gt_annos = copy.deepcopy([info['annos'] for info in self.data_infos])
        if eval_on_crop:
            gt_annos = self.crop_gt(gt_annos, eval_on_crop)
        ap_result_str, ap_dict = kitti_eval(
            gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
        print_log('\n' + ap_result_str, logger=logger)
        return ap_dict
    
    def crop_gt(self, gt_annos, eval_on_crop):
        '''
        Crop the ground trouth equivalent to the training time pre-processing.
        '''
        offset_w = eval_on_crop['offset_w']
        offset_h = eval_on_crop['offset_h']
        img_shape = eval_on_crop['img_shape']
        thresh_in_frame = 0.
        if 'thresh_in_frame' in eval_on_crop:
            thresh_in_frame = eval_on_crop['thresh_in_frame']
        print(f'evaluating on crop {offset_h},{offset_w} with img_shape {img_shape}')
        for gt_anno in gt_annos:
            pre_width = gt_anno['bbox'][:, 2] - gt_anno['bbox'][:, 0]
            pre_heihgt = gt_anno['bbox'][:, 3] - gt_anno['bbox'][:, 1]
            pre_area = pre_width * pre_heihgt
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = gt_anno['bbox'] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]-1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]-1)
            gt_anno['bbox'] = bboxes
            post_width = bboxes[:, 2] - bboxes[:, 0]
            post_height = bboxes[:, 3] - bboxes[:, 1]
            post_area = post_width * post_height
            in_frame = post_area / pre_area
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1]) & \
                (in_frame > thresh_in_frame)
            gt_anno['name'][~valid_inds] = 'ignore'
        return gt_annos