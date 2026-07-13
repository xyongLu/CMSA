# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import cv2
import numpy as np
from torchvision.transforms import functional as F


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


# def build_transforms(cfg, is_train=True):
#     assert is_train is True, 'Please only use build_transforms for training.'
#     assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
#     if is_train:
#         max_rotation = cfg.DATASET.MAX_ROTATION
#         min_scale = cfg.DATASET.MIN_SCALE
#         max_scale = cfg.DATASET.MAX_SCALE
#         max_translate = cfg.DATASET.MAX_TRANSLATE
#         input_size = cfg.DATASET.INPUT_SIZE
#         output_size = cfg.DATASET.OUTPUT_SIZE
#         flip = cfg.DATASET.FLIP
#         scale_type = cfg.DATASET.SCALE_TYPE
#     else:
#         scale_type = cfg.DATASET.SCALE_TYPE
#         max_rotation = 0
#         min_scale = 1
#         max_scale = 1
#         max_translate = 0
#         input_size = 512
#         output_size = [128]
#         flip = 0

#     # coco_flip_index = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
#     # if cfg.DATASET.WITH_CENTER:
#         # coco_flip_index.append(17)
#     if 'coco' in cfg.DATASET.DATASET:
#         dataset_name = 'COCO'
#     # elif 'crowd_pose' in cfg.DATASET.DATASET:
#     #     dataset_name = 'CROWDPOSE'
#     else:
#         raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
#     if cfg.DATASET.WITH_CENTER:
#         coco_flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER']
#     else:
#         coco_flip_index = FLIP_CONFIG[dataset_name]

#     transforms = Compose(
#         [
#             RandomAffineTransform(
#                 input_size,
#                 output_size,
#                 max_rotation,
#                 min_scale,
#                 max_scale,
#                 scale_type,
#                 max_translate,
#                 scale_aware_sigma=cfg.DATASET.SCALE_AWARE_SIGMA
#             ),
#             RandomHorizontalFlip(coco_flip_index, output_size, flip),
#             ToTensor(),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]
#     )
#     return transforms


# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, mask, joints):
#         for t in self.transforms:
#             image, mask, joints = t(image, mask, joints)
#         return image, mask, joints

#     def __repr__(self):
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms:
#             format_string += "\n"
#             format_string += "    {0}".format(t)
#         format_string += "\n)"
#         return format_string


class ToTensor(object):
    def __call__(self, image, mask, joints):
        return F.to_tensor(image), mask, joints


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints


class RandomHorizontalFlip(object):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    def __call__(self, image, mask, joints):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.prob:
            image = image[:, ::-1] - np.zeros_like(image)
            for i, _output_size in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = _output_size - joints[i][:, :, 0] - 1

        return image, mask, joints


class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate,
                 scale_aware_sigma=False):
        self.input_size = input_size
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        self.scale_aware_sigma = scale_aware_sigma

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long':
            scale = max(height, width)/200
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        for i, _output_size in enumerate(self.output_size):
            mat_output = self._get_affine_matrix(
                center, scale, (_output_size, _output_size), aug_rot
            )[:2]
            mask[i] = cv2.warpAffine(
                (mask[i]*255).astype(np.uint8), mat_output,
                (_output_size, _output_size)
            ) / 255
            mask[i] = (mask[i] > 0.5).astype(np.float32)

            joints[i][:, :, 0:2] = self._affine_joints(
                joints[i][:, :, 0:2], mat_output
            )
            if self.scale_aware_sigma:
                joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale

        mat_input = self._get_affine_matrix(
            center, scale, (self.input_size, self.input_size), aug_rot
        )[:2]
        image = cv2.warpAffine(
            image, mat_input, (self.input_size, self.input_size)
        )

        return image, mask, joints

## transforms for inference
def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


# def crop(img, center, scale, output_size, rot=0):
#     trans = get_affine_transform(center, scale, rot, output_size)

#     dst_img = cv2.warpAffine(img,
#                              trans,
#                              (int(output_size[0]), int(output_size[1])),
#                              flags=cv2.INTER_LINEAR)

#     return dst_img


def resize(image, input_size):
    h, w, _ = image.shape

    center = np.array([int(w/2.0+0.5), int(h/2.0+0.5)])
    if w < h:
        w_resized = input_size
        h_resized = int((input_size / w * h + 63) // 64 * 64)
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = input_size
        w_resized = int((input_size / h * w + 63) // 64 * 64)
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    scale = np.array([scale_w, scale_h])
    trans = get_affine_transform(center, scale, 0, (w_resized, h_resized))

    image_resized = cv2.warpAffine(
        image,
        trans,
        (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale


