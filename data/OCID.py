'''
TableTop Object Dataset Dataloader (Modified from scannetv2_inst.py)
'''

import os, sys
import open3d
import glob
import math

import numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops


def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation


class Dataset:
    def __init__(self):
        self.data_root = cfg.data_root

        self.batch_size = cfg.batch_size
        self.workers = cfg.test_workers

        self.subsample_factor = cfg.subsample_factor

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.mode = cfg.mode
        self.max_npoint = cfg.max_npoint

    def testLoader(self):
        with open(os.path.join(self.data_root, 'pcd_files.txt'), 'r') as f:
            self.pcd_files = [x.strip() for x in f.readlines()]

        logger.info('Test samples: {}'.format(len(self.pcd_files)))

        self.test_data_loader = DataLoader(list(range(len(self.pcd_files))),
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.workers,
                                           shuffle=False,
                                           sampler=None, 
                                           drop_last=False,
                                           pin_memory=True)


    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = max(int(instance_label.max()) + 1, 0)
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


    def process_rgb(self, point_cloud):
        """ Process RGB image
        """

        # Fill in missing pixel values
        num_missing = 480*640 - np.asarray(point_cloud.colors).shape[0]
        filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
        rgb_img = np.round(255 * filled_in_rgb_img.reshape(480,640,3)).astype(np.uint8)

        # Standardize RGB image
        rgb_img = self.standardize_image(rgb_img)

        return rgb_img        


    def process_depth(self, point_cloud):
        """ Process point cloud to get xyz image
        """

        # Fill in missing xyz values
        num_missing = 480*640 - np.asarray(point_cloud.points).shape[0]
        filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])

        xyz_img = np.asarray(filled_in_points).reshape(480,640,3)
        xyz_img[np.isnan(xyz_img)] = 0
        xyz_img[...,1] *= -1

        return xyz_img


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


    def load_data(self, pcd_filename):

        # Load point cloud
        point_cloud = open3d.io.read_point_cloud(pcd_filename, remove_nan_points=False)

        # Get RGB image
        rgb_img = self.process_rgb(point_cloud) # Shape: [H x W x 3]

        # Get XYZ image
        xyz_img = self.process_depth(point_cloud) # Shape: [H x W x 3]

        # Subsample
        rgb_img = rgb_img[::self.subsample_factor, ::self.subsample_factor]
        xyz_img = xyz_img[::self.subsample_factor, ::self.subsample_factor]

        # Reshape
        rgb = rgb_img.reshape(-1,3)
        xyz = xyz_img.reshape(-1,3)

        # Get rid of depth holes (0's), and stuff that is just too far away or too close
        depth_img = xyz_img[..., 2].reshape(-1)
        valid_mask = depth_img != 0
        valid_mask = np.logical_and(valid_mask, depth_img < cfg.far_plane)
        valid_mask = np.logical_and(valid_mask, depth_img > cfg.near_plane)
        rgb = rgb[valid_mask]
        xyz = xyz[valid_mask]

        return xyz, rgb, valid_mask


    def collate_fn(self, id):
        """
        :param id: a list of integers
        """
        locs = []
        locs_float = []
        feats = []
        valid_masks = []
        label_abs_paths = []
        batch_offsets = [0]

        for i, idx in enumerate(id):

            # Load data
            pcd_filename = self.pcd_files[idx]
            temp_idx = pcd_filename.split('/').index('OCID-dataset') # parse something like this: /data/OCID-dataset/YCB10/table/top/curved/seq36/pcd/result_2018-08-24-15-13-13.pcd
            label_abs_path = '/'.join(pcd_filename.split('/')[temp_idx+1:])
            xyz, rgb, valid_mask = self.load_data(pcd_filename)

            ### scale
            xyz_scaled = xyz * self.scale

            ### offset and crop
            xyz_scaled -= xyz_scaled.min(0)
            xyz_scaled, valid_idxs = self.crop(xyz_scaled)
            # NOTE: for TOD, this doesn't actually crop since I set cfg.max_npoint > #pixels

            xyz_scaled = xyz_scaled[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            if not np.all(valid_idxs):
                raise Exception("valid_idxs is not all True...")

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz_scaled.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz_scaled.shape[0], 1).fill_(i), torch.from_numpy(xyz_scaled).long()], 1))  # Used for voxelization
            locs_float.append(torch.from_numpy(xyz))  # XYZ coords used in the network
            feats.append(torch.from_numpy(rgb))  # Can set this to be torch.zeros((N,0), dtype=torch.float) if I don't want RGB
            valid_masks.append(torch.from_numpy(valid_mask))
            label_abs_paths.append(label_abs_path)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)
        valid_mask = torch.cat(valid_masks, 0).bool()             # bool (N)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'valid_mask': valid_mask,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                'label_abs_path' : label_abs_paths}
