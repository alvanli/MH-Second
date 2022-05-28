from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torch_data

from scipy.special import gamma
from scipy.integrate import trapz
import PyMieScatt as ps

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from ..utils import common_utils, box_utils, self_training_utils
from ..ops.roiaware_pool3d import roiaware_pool3d_utils


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
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
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

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
    @staticmethod
    def __vis__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=False):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def __vis_fake__(points, gt_boxes, ref_boxes=None, scores=None, use_fakelidar=True):
        import visual_utils.visualize_utils as vis
        import mayavi.mlab as mlab
        gt_boxes = copy.deepcopy(gt_boxes)
        if use_fakelidar:
            gt_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(gt_boxes)

        if ref_boxes is not None:
            ref_boxes = copy.deepcopy(ref_boxes)
            if use_fakelidar:
                ref_boxes = box_utils.boxes3d_kitti_lidar_to_fakelidar(ref_boxes)

        vis.draw_scenes(points, gt_boxes, ref_boxes=ref_boxes, ref_scores=scores)
        mlab.show(stop=True)

    @staticmethod
    def extract_fov_data(points, fov_degree, heading_angle):
        """
        Args:
            points: (N, 3 + C)
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        points_new = common_utils.rotate_points_along_z(
            points.copy()[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        angle = np.arctan2(points_new[:, 1], points_new[:, 0])
        fov_mask = ((np.abs(angle) < half_fov_degree) & (points_new[:, 0] > 0))
        points = points_new[fov_mask]
        return points

    @staticmethod
    def extract_fov_gt(gt_boxes, fov_degree, heading_angle):
        """
        Args:
            anno_dict:
            fov_degree: [0~180]
            heading_angle: [0~360] in lidar coords, 0 is the x-axis, increase clockwise
        Returns:
        """
        half_fov_degree = fov_degree / 180 * np.pi / 2
        heading_angle = -heading_angle / 180 * np.pi
        gt_boxes_lidar = copy.deepcopy(gt_boxes)
        gt_boxes_lidar = common_utils.rotate_points_along_z(
            gt_boxes_lidar[np.newaxis, :, :], np.array([heading_angle])
        )[0]
        gt_boxes_lidar[:, 6] += heading_angle
        gt_angle = np.arctan2(gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0])
        fov_gt_mask = ((np.abs(gt_angle) < half_fov_degree) & (gt_boxes_lidar[:, 0] > 0))
        return fov_gt_mask

    def fill_pseudo_labels(self, input_dict):
        gt_boxes = self_training_utils.load_ps_label(input_dict['frame_id'])
        gt_scores = gt_boxes[:, 8]
        gt_classes = gt_boxes[:, 7]
        gt_boxes = gt_boxes[:, :7]

        # only suitable for only one classes, generating gt_names for prepare data
        gt_names = np.array([self.class_names[0] for n in gt_boxes])

        input_dict['gt_boxes'] = gt_boxes
        input_dict['gt_names'] = gt_names
        input_dict['gt_classes'] = gt_classes
        input_dict['gt_scores'] = gt_scores
        input_dict['pos_ps_bbox'] = (gt_classes > 0).sum()
        input_dict['ign_ps_bbox'] = gt_boxes.shape[0] - input_dict['pos_ps_bbox']
        input_dict.pop('num_points_in_gt', None)


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
                points: optional, (N, 3 + C_in)
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
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        # 'points', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points'
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
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
                elif key in ['gt_scores']:
                    max_gt = max([len(x) for x in val])
                    batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                    for k in range(batch_size):
                        batch_scores[k, :val[k].__len__()] = val[k]
                    ret[key] = batch_scores
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def eval(self):
        self.training = False
        self.data_processor.eval()

    def train(self):
        self.training = True
        self.data_processor.train()


class LISA():
    def __init__(self,m=1.328,lam=905,rmax=200,rmin=1.5,bdiv=3e-3,dst=0.05,
                 dR=0.09,saved_model=False,atm_model='rain',mode='strongest'):
        '''
        Initialize LISA class
        Parameters
        ----------
        m           : refractive index contrast
        lam         : wavelength (nm)
        rmax        : max lidar range (m)
        rmin        : min lidar range (m)
        bdiv        : beam divergence angle (rad)
        dst         : droplet diameter starting point (mm)
        dR          : range accuracy (m)
        saved_model : use saved mie coefficients (bool)
        atm_model   : atmospheric model type
        mode        : lidar return mode: "strongest" or "last"

        Returns
        -------
        None.

        '''
        self.m    = m
        self.lam  = lam
        self.rmax = rmax   # max range (m)
        self.bdiv = bdiv  # beam divergence (rad)
        self.dst  = dst   # min rain drop diameter to be sampled (mm)
        self.rmin = rmin   # min lidar range (bistatic)
        self.dR   = dR
        self.mode = mode
        self.atm_model = atm_model
        
        self.base_path = "/home/OpenPCDet/pcdet/datasets/"
        if saved_model:
            # If Mie parameters are saved, use those
            dat = np.load(self.base_path + 'mie_q.npz')
            self.D     = dat['D']
            self.qext  = dat['qext']
            self.qback = dat['qback']
        else:
            try:
                dat = np.load(self.base_path + 'mie_q.npz')
                self.D     = dat['D']
                self.qext  = dat['qext']
                self.qback = dat['qback']
            except:
                # else calculate Mie parameters
                print('Calculating Mie coefficients... \nThis might take a few minutes')
                self.D,self.qext,self.qback = self.calc_Mie_params()
                print('Mie calculation done...')
        
        # Diameter distribution function based on user input
        if atm_model=='rain':
            self.N_model = lambda D, Rr    : self.N_MP_rain(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MP_tot_rain(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MP_Sample_rain(Rr,N,dst)
            
            # Augmenting function: hybrid Monte Carlo
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        
        elif atm_model=='snow':
            self.N_model = lambda D, Rr    : self.N_MG_snow(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MG_tot_snow(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MG_Sample_snow(Rr,N,dst)
            self.m       = 1.3031 # refractive index of ice
            
            # Augmenting function: hybrid Monte Carlo
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        
        elif atm_model=='chu_hogg_fog':
            self.N_model = lambda D : self.Nd_chu_hogg(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)
        
        elif atm_model=='strong_advection_fog':
            self.N_model = lambda D : self.Nd_strong_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)
        
        elif atm_model=='moderate_advection_fog':
            self.N_model = lambda D : self.Nd_moderate_advection_fog(D)
            
            # Augmenting function: average effects
            self.augment  = lambda pc : self.augment_avg(pc)
    
    def augment_mc(self,pc,Rr):
        '''
        Augment clean pointcloud for a given rain rate
        Parameters
        ----------
        pc : pointcloud (N,4) -> x,y,z,reflectivity
        Rr : rain rate (mm/hr)

        Returns
        -------
        pc_new : new noisy point cloud (N,5) -> x,y,z,reflectivity,label
                        label 0 -> lost point
                        label 1 -> randomly scattered point
                        label 2 -> not-scattered 
        '''
        shp    = pc.shape
        pc_new = np.zeros((shp[0],shp[1]+1))
        leng = len(pc)
        for i in range(leng):
            x    = pc[i,0]
            y    = pc[i,1]
            z    = pc[i,2]
            ref  = pc[i,3]
            if ref!=0:
                pc_new[i,:]  = self.lisa_mc(x,y,z,ref,Rr)            
        return pc_new
    
    def lisa_mc(self,x,y,z,ref,Rr):
        '''
        For a single lidar return, performs a hybrid Monte-Carlo experiment

        Parameters
        ----------
        x,y,z : coordinates of the point
        ref   : reflectivity [0 1]
        Rr    : rain rate (mm/hr)

        Returns
        -------
        x,y,z   : new coordinates of the noisy lidar point
        ref_new : new reflectivity
        '''
        rmax = self.rmax                      # max range (m)
        Pmin = 0.9*rmax**(-2)                 # min measurable power (arb units)
        
        bdiv = self.bdiv                      # beam divergence (rad)
        Db   = lambda x: 1e3*np.tan(bdiv)*x   # beam diameter (mm) for a given range (m)
        
        dst  = self.dst                       # min rain drop diameter to be sampled (mm)
        n    = self.m                         # refractive index of scatterer
        rmin = self.rmin                      # min lidar range (bistatic)
        
        
        Nd          = self.N_model(self.D,Rr) # density of rain droplets (m^-3)
        alpha, beta = self.alpha_beta(Nd)     # extinction coeff. (1/m)  
        
        ran   = np.sqrt(x**2 + y**2 + z**2)                               # range in m
        if ran>rmin:
            bvol  = (np.pi/3)*ran*(1e-3*Db(ran)/2)**2                         # beam volume in m^3 (cone)
            Nt    = self.N_tot(Rr,dst) * bvol                                 # total number of particles in beam path
            Nt    = np.int32(np.floor(Nt) + (np.random.rand() < Nt-int(Nt)))  # convert to integer w/ probabilistic rounding
        else:
            Nt = 0
            
        ran_r = ran*(np.random.rand(Nt))**(1/3) # sample distances from a quadratic pdf
        indx  = np.where(ran_r>rmin)[0]         # keep points where ranges larger than rmin
        Nt    = len(indx)                       # new particle number
        
        P0  = ref*np.exp(-2*alpha*ran)/(ran**2) # power
        snr = P0/Pmin # signal noise ratio
        if Nt>0:
            Dr    = self.N_sam(Rr,Nt,dst) # randomly sample Nt particle diameters
            ref_r = abs((n-1)/(n+1))**2   # Fresnel reflection at normal incidence
            ran_r = ran_r[indx]
            
            # Calculate powers for all particles       
            Pr = ref_r*np.exp(-2*alpha*ran_r)*np.minimum((Dr/Db(ran_r))**2,np.ones(Dr.shape))/(ran_r**2)
            if (self.mode=='strongest'):
                ind_r = np.argmax(Pr) # index of the max power
                
                if P0<Pmin and Pr[ind_r]<Pmin: # if all smaller than Pmin, do nothing
                    ran_new = 0
                    ref_new = 0
                    labl    = 0 # label for lost point
                elif P0<Pr[ind_r]: # scatterer has larger power
                    ran_new = ran_r[ind_r] # new range is scatterer range
                    ref_new = ref_r*np.exp(-2*alpha*ran_new)*np.minimum((Dr[ind_r]/Db(ran_r[ind_r]))**2,1) # new reflectance biased by scattering
                    labl    = 1 # label for randomly scattered point 
                else: # object return has larger power
                    sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                    ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                    ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                    labl    = 2                             # label for a non-scattering point
            elif (self.mode=='last'):
                # if object power larger than Pmin, then nothing is scattered
                if P0>Pmin:
                    sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                    ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                    ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                    labl    = 2                             # label for a non-scattering point
                # otherwise find the furthest point above Pmin
                else:
                    inds = np.where(Pr>Pmin)[0]
                    if len(inds) == 0:
                        ran_new = 0
                        ref_new = 0
                        labl    = 0 # label for lost point
                    else:
                        ind_r   = np.where(ran_r == np.max(ran_r[inds]))[0]
                        ran_new = ran_r[ind_r] # new range is scatterer range
                        ref_new = ref_r*np.exp(-2*alpha*ran_new)*np.minimum((Dr[ind_r]/Db(ran_r[ind_r]))**2,1) # new reflectance biased by scattering
                        labl    = 1 # label for randomly scattered point 
                    
            else:
                print("Invalid lidar return mode")
            
        else:
            if P0<Pmin:
                ran_new = 0
                ref_new = 0
                labl    = 0 # label for lost point
            else:
                sig     = self.dR/np.sqrt(2*snr)        # std of range uncertainty
                ran_new = ran + np.random.normal(0,sig) # range with uncertainty added
                ref_new = ref*np.exp(-2*alpha*ran)      # new reflectance modified by scattering
                labl    = 2                             # label for a non-scattering point
        
        # Angles are same
        if ran>0:
            phi = np.arctan2(y,x)  # angle in radians
            the = np.arccos(z/ran) # angle in radians
        else:
            phi,the=0,0
        
        # Update new x,y,z based on new range
        x = ran_new*np.sin(the)*np.cos(phi)
        y = ran_new*np.sin(the)*np.sin(phi)
        z = ran_new*np.cos(the)
        
        return x,y,z,ref_new,labl
    
    def augment_avg(self,pc):

        shp    = pc.shape      # data shape
        pc_new = np.zeros(shp) # init new point cloud
        leng   = shp[0]        # data length
        
        # Rename variables for better readability
        x    = pc[:,0]
        y    = pc[:,1]
        z    = pc[:,2]
        ref  = pc[:,3]          
        
        # Get parameters from class init
        rmax = self.rmax       # max range (m)
        Pmin = 0.9*rmax**(-2)  # min measurable power (arb units)
        rmin = self.rmin       # min lidar range (bistatic)
        
        # Calculate extinction coefficient from the particle distribution
        Nd          = self.N_model(self.D) # density of rain droplets (m^-3)
        alpha, beta = self.alpha_beta(Nd)  # extinction coeff. (1/m)  
        
        ran   = np.sqrt(x**2 + y**2 + z**2)  # range in m
        indx  = np.where(ran>rmin)[0]         # keep points where ranges larger than rmin
        
        P0        = np.zeros((leng,))                                  # init back reflected power
        P0[indx]  = ref[indx]*np.exp(-2*alpha*ran[indx])/(ran[indx]**2) # calculate reflected power
        snr       = P0/Pmin                                             # signal noise ratio
        
        indp = np.where(P0>Pmin)[0] # keep points where power is larger than Pmin
        
        sig        = np.zeros((leng,))                         # init sigma - std of range uncertainty
        sig[indp]  = self.dR/np.sqrt(2*snr[indp])               # calc. std of range uncertainty
        ran_new    = np.zeros((leng,))                         # init new range
        ran_new[indp]    = ran[indp] + np.random.normal(0,sig[indp])  # range with uncertainty added, keep range 0 if P<Pmin
        ref_new    = ref*np.exp(-2*alpha*ran)                   # new reflectance modified by scattering
        
        # Init angles
        phi = np.zeros((leng,))
        the = np.zeros((leng,))
        
        phi[indx] = np.arctan2(y[indx],x[indx])   # angle in radians
        the[indx] = np.arccos(z[indx]/ran[indx])  # angle in radians
        
        # Update new x,y,z based on new range
        pc_new[:,0] = ran_new*np.sin(the)*np.cos(phi)
        pc_new[:,1] = ran_new*np.sin(the)*np.sin(phi)
        pc_new[:,2] = ran_new*np.cos(the)
        pc_new[:,3] = ref_new
        
        return pc_new
    def msu_rain(self,pc,Rr):
        '''
        Lidar rain simulator from Goodin et al., 'Predicting the Influence of 
        Rain on LIDAR in ADAS', electronics 2019

        Parameters
        ----------
        pc : point cloud (N,4)
        Rr : rain rate in mm/hr

        Returns
        -------
        pc_new : output point cloud (N,4)

        '''
        shp    = pc.shape      # data shape
        pc_new = np.zeros(shp) # init new point cloud
        leng   = shp[0]        # data length
        
        # Rename variables for better readability
        x    = pc[:,0]
        y    = pc[:,1]
        z    = pc[:,2]
        ref  = pc[:,3]          
        
        # Get parameters from class init
        rmax = self.rmax       # max range (m)
        Pmin = 0.9*rmax**(-2)/np.pi  # min measurable power (arb units)
        
        # Calculate extinction coefficient from rain rate
        alpha = 0.01* Rr**0.6
        
        ran      = np.sqrt(x**2 + y**2 + z**2)  # range in m
        indv     = np.where(ran>0)[0] # clean data might already have invalid points
        P0       = np.zeros((leng,))
        P0[indv] = ref[indv]*np.exp(-2*alpha*ran[indv])/(ran[indv]**2) # calculate reflected power
        
        # init new ref and ran
        ran_new = np.zeros((leng,))
        ref_new = np.zeros((leng,))
        
        indp = np.where(P0>Pmin)[0] # points where power is greater than Pmin
        ref_new[indp] = ref[indp]*np.exp(-2*alpha*ran[indp]) # reflectivity reduced by atten
        sig = 0.02*ran[indp]* (1-np.exp(-Rr))**2
        ran_new[indp] = ran[indp] + np.random.normal(0,sig) # new range with uncertainty
        
        # Init angles
        phi = np.zeros((leng,))
        the = np.zeros((leng,))
        
        phi[indp] = np.arctan2(y[indp],x[indp])   # angle in radians
        the[indp] = np.arccos(z[indp]/ran[indp])  # angle in radians
        
        # Update new x,y,z based on new range
        pc_new[:,0] = ran_new*np.sin(the)*np.cos(phi)
        pc_new[:,1] = ran_new*np.sin(the)*np.sin(phi)
        pc_new[:,2] = ran_new*np.cos(the)
        pc_new[:,3] = ref_new
        
        return pc_new
    def haze_point_cloud(self,pts_3D,Rr=0):
        '''
        Modified from
        https://github.com/princeton-computational-imaging/SeeingThroughFog/blob/master/tools/DatasetFoggification/lidar_foggification.py

        Parameters
        ----------
        pts_3D : Point cloud
        Rr : Rain rate (mm/hr)

        Returns
        -------
        dist_pts_3d : Augmented point cloud
        '''
        n = []
        #Velodyne HDL64S2
        n = 0.05
        g = 0.35
        dmin = 2
            
        d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
        detectable_points = np.where(d>dmin)
        d = d[detectable_points]
        pts_3D = pts_3D[detectable_points]
        
        #######################################################################
        # This is the main modified part
        # For comparison we would like to calculate the extinction coefficient
        # from rain rate instead of sampling it from a distribution
        if (self.atm_model == 'rain') or (self.atm_model == 'snow'):
        	Nd  = self.N_model(self.D,Rr) # density of water droplets (m^-3)
        elif (self.atm_model == 'chu_hogg_fog') or (self.atm_model=='strong_advection_fog') or (self.atm_model=='moderate_advection_fog'):
        	Nd  = self.N_model(self.D) # density of water droplets (m^-3)
        else:
        	print('Warning: weather model not implemented')
        alpha, beta = self.alpha_beta(Nd)     # extinction coeff. (1/m)
        #######################################################################
    
        beta_usefull = alpha*np.ones(d.shape) # beta is the extinction coefficient (actually alpha)
        dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
        dnew = -np.log(1 - 0.5) / (beta_usefull)
    
        probability_lost = 1 - np.exp(-beta_usefull*dmax)
        lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost
    
        cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
        random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
        idx_stable = np.where(d<dmax)[0]
        old_points = np.zeros((len(idx_stable), 5))
        old_points[:,0:4] = pts_3D[idx_stable,:]
        old_points[:,3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
        old_points[:, 4] = np.zeros(np.shape(old_points[:,3]))
    
        cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
        cloud_scatter = np.zeros((len(cloud_scatter_idx), 5))
        cloud_scatter[:,0:4] =  pts_3D[cloud_scatter_idx,:]
        cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
        cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
        cloud_scatter[:, 4] = np.ones(np.shape(cloud_scatter[:, 3]))
    
    
        # Subsample random scatter abhaengig vom noise im Lidar
        random_scatter_idx = np.where(random_scatter)[0]
        scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
        drand = np.random.uniform(high=scatter_max[random_scatter_idx])
        # scatter outside min detection range and do some subsampling. Not all points are randomly scattered.
        # Fraction of 0.05 is found empirically.
        drand_idx = np.where(drand>dmin)
        drand = drand[drand_idx]
        random_scatter_idx = random_scatter_idx[drand_idx]
        # Subsample random scattered points to 0.05%
        fraction_random = .05 # just set this according to the comment above^ rather than parsing arguments; also probably .05 not .05%
        subsampled_idx = np.random.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
        drand = drand[subsampled_idx]
        random_scatter_idx = random_scatter_idx[subsampled_idx]
    
    
        random_scatter = np.zeros((len(random_scatter_idx), 5))
        random_scatter[:,0:4] = pts_3D[random_scatter_idx,:]
        random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
        random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
        random_scatter[:, 4] = 2*np.ones(np.shape(random_scatter[:, 3]))
    
        dist_pts_3d = np.concatenate((old_points, cloud_scatter,random_scatter), axis=0)
    
        return dist_pts_3d
    
    def calc_Mie_params(self):
        '''
        Calculate scattering efficiencies
        Returns
        -------
        D     : Particle diameter (mm)
        qext  : Extinction efficiency
        qback : Backscattering efficiency

        '''
        out   = ps.MieQ_withDiameterRange(self.m, self.lam, diameterRange=(1,1e7),
                                        nd=2000, logD=True)
        D     = out[0]*1e-6
        qext  = out[1]
        qback = out[6]
        
        # Save for later use since this function takes long to run
        np.savez('mie_q.npz',D=D,qext=qext,qback=qback)
        
        return D,qext,qback
    
    
    def alpha_beta(self,Nd):
        '''
        Calculates extunction and backscattering coefficients
        Parameters
        ----------
        Nd : particle size distribution, m^-3 mm^-1

        Returns
        -------
        alpha : extinction coefficient
        beta  : backscattering coefficient
        '''
        D  = self.D
        qe = self.qext
        qb = self.qback
        alpha = 1e-6*trapz(D**2*qe*Nd,D)*np.pi/4 # m^-1
        beta  = 1e-6*trapz(D**2*qb*Nd,D)*np.pi/4 # m^-1
        return alpha, beta
    
    # RAIN
    def N_MP_rain(self,D,Rr):
        '''
        Marshall - Palmer rain model

        Parameters
        ----------
        D  : rain droplet diameter (mm)
        Rr : rain rate (mm h^-1)

        Returns
        -------
        number of rain droplets for a given diameter (m^-3 mm^-1)
        '''
        return 8000*np.exp(-4.1*Rr**(-0.21)*D)
    
    def N_MP_tot_rain(self,Rr,dstart):
        '''
        Integrated Marshall - Palmer Rain model

        Parameters
        ----------
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        Returns
        -------
        rain droplet density (m^-3) for a given min diameter
        '''
        lam = 4.1*Rr**(-0.21)
        return 8000*np.exp(-lam*dstart)/lam

    def MP_Sample_rain(self,Rr,N,dstart):
        '''
        Sample particle diameters from Marshall Palmer distribution

        Parameters
        ----------
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        Returns
        -------
        diameters : diameter of the samples

        '''
        lmda      = 4.1*Rr**(-0.21)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters
    
    # SNOW
    def N_MG_snow(self,D,Rr):
        '''
        Marshall - Palmer snow model

        Parameters
        ----------
        D  : snow diameter (mm)
        Rr : water equivalent rain rate (mm h^-1)

        Returns
        -------
        number of snow particles for a given diameter (m^-3 mm^-1)
        '''
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*D)
    
    def N_MG_tot_snow(self,Rr,dstart):
        '''
        Integrated Marshall - Gunn snow model

        Parameters
        ----------
        Rr     : rain rate (mm h^-1)
        dstart : integral starting point for diameter (mm)

        Returns
        -------
        snow particle density (m^-3) for a given min diameter
        '''
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*dstart)/lmda

    def MG_Sample_snow(self,Rr,N,dstart):
        '''
        Sample particle diameters from Marshall Palmer distribution

        Parameters
        ----------
        Rr     : rain rate (mm/hr)
        N      : number of samples
        dstart : Starting diameter (min diameter sampled)

        Returns
        -------
        diameters : diameter of the samples

        '''
        lmda      = 2.55* Rr**(-0.48)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters
    # FOG
    def N_GD(self,D,rho,alpha,g,Rc):
        '''
        Gamma distribution model
        Note the parameters are NOT normalized to unitless values
        For example D^alpha term will have units Length^alpha
        It is therefore important to use exactly the same units for D as those
        cited in the paper by Rasshofer et al. and then perform unit conversion
        after an N(D) curve is generated
    
        D  : rain diameter
        Outputs number of rain droplets for a given diameter
        '''
        b = alpha/(g*Rc**g)
        
        Nd = g*rho*b**((alpha+1)/g)*(D/2)**alpha*np.exp(-b*(D/2)**g)/gamma((alpha+1)/g)
        
        return Nd
    # Coastal fog distribution
    # With given parameters, output has units cm^-3 um^-1 which is
    # then converted to m^-3 mm^-1 which is what alpha_beta() expects
    # so whole quantity is multiplied by (100 cm/m)^3 (1000 um/mm)
    def Nd_haze_coast(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=1,g=0.5,Rc=0.05e-3)
    
    # Continental fog distribution
    def Nd_haze_continental(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=2,g=0.5,Rc=0.07)
    
    # Strong advection fog
    def Nd_strong_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=10)
    
    # Moderate advection fog
    def Nd_moderate_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=8)
    
    # Strong spray
    def Nd_strong_spray(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=4)
    
    # Moderate spray
    def Nd_moderate_spray(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=2)
    
    # Chu/Hogg
    def Nd_chu_hogg(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=2,g=0.5,Rc=1)
    