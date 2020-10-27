'''
TableTop Object Dataset Dataloader (Modified from scannetv2_inst.py)
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops


def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.
        Assumes camera uses left-handed coordinate system, with 
            x-axis pointing right
            y-axis pointing up
            z-axis pointing "forward"

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = np.indices((camera_params['img_height'], camera_params['img_width']),
                         dtype=np.float32).transpose(1,2,0)
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]

    return xyz_img


def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation


class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.mode = cfg.mode
        self.max_npoint = cfg.max_npoint

        self.camera_params = {
            'img_width' : 640, 
            'img_height' : 480,
            'near' : 0.01,
            'far' : 100,
            'fov' : 45, # vertical field of view in angles
        }

        # Data augmentation params
        self.gamma_shape = cfg.gamma_shape
        self.gamma_scale = cfg.gamma_scale
        self.gaussian_scale_range = cfg.gaussian_scale_range
        self.gp_rescale_factor_range = cfg.gp_rescale_factor_range

    def trainLoader(self):
        with open(os.path.join(self.data_root, 'label_files.txt'), 'r') as f:
            self.label_filenames = f.readlines()
        # Only look at examples with table in it.
        self.label_filenames = [x for x in self.label_filenames if 'segmentation_00000.png' not in x]

        logger.info('Training samples: {}'.format(len(self.label_filenames)))

        self.train_data_loader = DataLoader(list(range(len(self.label_filenames))),
                                            batch_size=self.batch_size,
                                            collate_fn=self.train_collate_fn,
                                            num_workers=self.train_workers,
                                            shuffle=True,
                                            sampler=None, 
                                            drop_last=True,
                                            pin_memory=True)


    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def add_noise_to_depth(self, depth_img):
        """ Add noise to depth image. 
            This is adapted from the DexNet 2.0 code.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Multiplicative noise: Gamma random variable
        multiplicative_noise = np.random.gamma(self.gamma_shape, self.gamma_scale)
        depth_img = multiplicative_noise * depth_img

        return depth_img


    def add_noise_to_xyz(self, xyz_img, depth_img):
        """ Add (approximate) Gaussian Process noise to ordered point cloud

            @param xyz_img: a [H x W x 3] ordered point cloud
        """
        xyz_img = xyz_img.copy()
        H, W, C = xyz_img.shape

        # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
        #                 which is rescaled with bicubic interpolation.
        gp_rescale_factor = np.random.randint(self.gp_rescale_factor_range[0],
                                              self.gp_rescale_factor_range[1])
        gp_scale = np.random.uniform(self.gaussian_scale_range[0],
                                     self.gaussian_scale_range[1])

        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

        return xyz_img


    def standardize_image(self, image):
        """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes

            @return: a [H x W x 3] numpy array of np.float32
        """
        image_standardized = np.zeros_like(image).astype(np.float32)

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for i in range(3):
            image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

        return image_standardized


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def process_instance_label(self, instance_label):
        """Map labels from [0, 1, ..., K+1] to {-100, 0, 1, ..., K-1}.

        Also return segmentation labels, which is just a foreground category.

        Args:
            instance_label: [H, W] numpy.ndarray
        """
        new_instance_labels = np.ones_like(instance_label) * cfg.ignore_label  # Assume cfg.ignore_label is -100
        semantic_labels = new_instance_labels.copy()

        curr_instance_label = 0
        for i in range(instance_label.max()+1):
            mask = instance_label == i
            if len(np.where(mask)[0]) > 0:
                if i in [0, 1]:  # background / table
                    new_instance_labels[mask] = cfg.ignore_label
                    semantic_labels[mask] = i
                else:
                    new_instance_labels[mask] = curr_instance_label
                    curr_instance_label += 1
                    semantic_labels[mask] = 2  # OBJECT label

        return semantic_labels, new_instance_labels


    def load_data(self, rgb_filename, depth_filename, segmentation_filename):

        # RGB image
        rgb_img = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.standardize_image(rgb_img)  # Same as UOIS-Net training

        # Depth image
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        depth_img = (depth_img / 1000.).astype(np.float32)  # millimeters to meters
        xyz_img = compute_xyz(depth_img, self.camera_params)
        xyz_img_augmented = compute_xyz(self.add_noise_to_depth(depth_img), self.camera_params)
        xyz_img_augmented = self.add_noise_to_xyz(xyz_img_augmented, depth_img)
        # the noise addition here is same as UOIS-Net training

        # Labels
        instance_labels = imread_indexed(segmentation_filename)
        segmentation_labels, instance_labels = self.process_instance_label(instance_labels)

        # Reshape
        rgb = rgb_img.reshape(-1,3)
        xyz = xyz_img.reshape(-1,3)
        xyz_augmented = xyz_img_augmented.reshape(-1,3)
        segmentation_labels = segmentation_labels.reshape(-1)
        instance_labels = instance_labels.reshape(-1)

        # Get rid of depth holes (0's), and stuff that is just too far away
        holes_mask = np.logical_or(depth_img.reshape(-1) == 0, depth_img.reshape(-1) > cfg.far_plane)
        rgb = rgb[~holes_mask]
        xyz = xyz[~holes_mask]
        xyz_augmented = xyz_augmented[~holes_mask]
        segmentation_labels = segmentation_labels[~holes_mask]
        instance_labels = instance_labels[~holes_mask]

        return xyz, xyz_augmented, rgb, segmentation_labels, instance_labels


    def train_collate_fn(self, id):
        """
        :param id: a list of integers
        """
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            # Load data
            label_filename = self.label_filenames[idx].strip()
            rgb_filename = label_filename.replace('segmentation', 'rgb').replace('png', 'jpeg')
            depth_filename = label_filename.replace('segmentation', 'depth')
            xyz_origin, xyz_aug, rgb, label, instance_label = \
                self.load_data(rgb_filename, depth_filename,label_filename)

            ### scale
            xyz = xyz_aug * self.scale

            ### offset and crop
            xyz -= xyz.min(0)
            xyz, valid_idxs = self.crop(xyz)

            xyz = xyz[valid_idxs]
            xyz_aug = xyz_aug[valid_idxs]
            xyz_origin = xyz_origin[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_origin, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list

            instance_label[np.where(instance_label != cfg.ignore_label)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))  # Used for voxelization
            locs_float.append(torch.from_numpy(xyz_aug))  # XYZ coords used in the network
            feats.append(torch.from_numpy(rgb))  # Can set this to be torch.zeros((N,0), dtype=torch.float) if I don't want RGB
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}



