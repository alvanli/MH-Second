import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import glob
import os

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils, self_training_utils
from ..dataset import DatasetTemplate


class WatoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = "/home/road/combined_cars/"
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.lidar_paths = glob.glob(self.root_path + "pc*.npy")

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def __len__(self):
        return len(self.lidar_paths)

    def __getitem__(self, index):
        points = np.load(self.lidar_paths[index])
        points = self.remove_ego_points(points)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': int(os.path.basename(self.lidar_paths[index]).split(".")[0][2:])
        }


        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        # if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
        #     gt_boxes = input_dict['gt_boxes']
        #     gt_boxes[np.isnan(gt_boxes)] = 0
        #     input_dict['gt_boxes'] = gt_boxes

        # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in input_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
        #     input_dict['gt_boxes'] = input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

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
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
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
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

