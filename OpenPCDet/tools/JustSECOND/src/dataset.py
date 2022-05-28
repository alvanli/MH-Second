from functools import partial
from collections import defaultdict

import numpy as np
import torch
import cumm.tensorview as tv
from .utils import mask_points_by_range

POINT_CLOUD_RANGE = np.array([-75.2, -75.2, -2, 75.2, 75.2, 4])
USE_LEAD_XYZ = True
USED_FEATURE_LIST = ["x", "y", "z"]
SRC_FEATURE_LIST = ["x", "y", "z"]
NUM_POINT_FEATURES = 3
MAX_POINTS_PER_VOXEL = 5
MAX_NUMBER_OF_VOXELS = 90000
VOXEL_SIZE = np.array([0.1, 0.1, 0.15])
GRID_SIZE = (POINT_CLOUD_RANGE[3:6] - POINT_CLOUD_RANGE[0:3]) / np.array(VOXEL_SIZE)
GRID_SIZE = np.round(GRID_SIZE).astype(np.int64)

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points



class PointCloudPrep():
    def __init__(self):        
        self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=VOXEL_SIZE,
                coors_range_xyz=POINT_CLOUD_RANGE,
                num_point_features=NUM_POINT_FEATURES,
                max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                max_num_voxels=MAX_NUMBER_OF_VOXELS,
            )

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(USED_FEATURE_LIST)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        # for x in self.used_feature_list:
        #     if x in ['x', 'y', 'z']:
        #         continue
        #     idx = self.src_feature_list.index(x)
        #     point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features

    def point_feature_encoder(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
        """
        points = self.remove_ego_points(data_dict["points"])
        data_dict['points'] = self.absolute_coordinates_encoding(points)
        return data_dict

    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def transform_points_to_voxels(self, data_dict=None):
        if data_dict is None:
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels)

        voxel_output = self.voxel_generator.generate(data_dict['points'])
        voxels, coordinates, num_points = voxel_output

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def mask_points_and_boxes_outside_range(self, data_dict=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range)

        mask = mask_points_by_range(data_dict['points'], POINT_CLOUD_RANGE)
        data_dict['points'] = data_dict['points'][mask]
        return data_dict

    def data_process(self, data_dict):
        data_dict = self.mask_points_and_boxes_outside_range(data_dict)
        data_dict = self.transform_points_to_voxels(data_dict)
        return data_dict

    def get_batch(self, sample):
        data_dict = defaultdict(list)
        for key, val in sample.items():
            data_dict[key].append(val)
        ret = {}

        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            else: # use_lead_xyz
                ret[key] = np.stack(val, axis=0)
        return ret
        
    
    def load_data_to_gpu(self, batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
        return batch_dict


    def prepare_data(self, points):
        input_dict = {
            'points': points,
            # 'frame_id': int(os.path.basename(self.lidar_paths[index]).split(".")[0][2:])
        }
        data_dict = self.point_feature_encoder(input_dict)
        data_dict = self.data_process(data_dict)
        data_dict = self.get_batch(data_dict)
    
        return self.load_data_to_gpu(data_dict)