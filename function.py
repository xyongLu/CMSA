import torch
import math
import numpy as np
import cv2

from transforms import FLIP_CONFIG
from utils import transform_preds

## evaluate
def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


## inference
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord

def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i,j] = dr[border: -border, border: -border].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
    return hm

def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals


##
def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):
    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list

def get_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    outputs = []
    heatmaps = []
    tags = []

    outputs.append(model(image))
    heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
    tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])

    if with_flip:
        outputs.append(model(torch.flip(image, [3])))
        outputs[-1] = torch.flip(outputs[-1], [3])
        heatmaps.append(outputs[-1][:, :cfg.DATASET.NUM_JOINTS])
        tags.append(outputs[-1][:, cfg.DATASET.NUM_JOINTS:])
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]
        heatmaps[-1] = heatmaps[-1][:, flip_index, :, :]
        if cfg.MODEL.TAG_PER_JOINT:
            tags[-1] = tags[-1][:, flip_index, :, :]

    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    return outputs, heatmaps, tags


def get_multi_stage_outputs(
        cfg, model, image, with_flip=False,
        project2image=False, size_projected=None
):
    # outputs = []
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    outputs = model(image)
    for i, output in enumerate(outputs):
        if len(outputs) > 1 and i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=False
            )

        offset_feat = cfg.DATASET.NUM_JOINTS \
            if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

        if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
            heatmaps_avg += output[:, :cfg.DATASET.NUM_JOINTS]
            num_heatmaps += 1

        if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
            tags.append(output[:, offset_feat:])

    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg/num_heatmaps)

    if with_flip:
        if 'coco' in cfg.DATASET.DATASET:
            dataset_name = 'COCO'
        elif 'crowd_pose' in cfg.DATASET.DATASET:
            dataset_name = 'CROWDPOSE'
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        flip_index = FLIP_CONFIG[dataset_name + '_WITH_CENTER'] \
            if cfg.DATASET.WITH_CENTER else FLIP_CONFIG[dataset_name]

        heatmaps_avg = 0
        num_heatmaps = 0
        outputs_flip = model(torch.flip(image, [3]))
        for i in range(len(outputs_flip)):
            output = outputs_flip[i]
            if len(outputs_flip) > 1 and i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            output = torch.flip(output, [3])
            outputs.append(output)

            offset_feat = cfg.DATASET.NUM_JOINTS \
                if cfg.LOSS.WITH_HEATMAPS_LOSS[i] else 0

            if cfg.LOSS.WITH_HEATMAPS_LOSS[i] and cfg.TEST.WITH_HEATMAPS[i]:
                heatmaps_avg += \
                    output[:, :cfg.DATASET.NUM_JOINTS][:, flip_index, :, :]
                num_heatmaps += 1

            if cfg.LOSS.WITH_AE_LOSS[i] and cfg.TEST.WITH_AE[i]:
                tags.append(output[:, offset_feat:])
                if cfg.MODEL.TAG_PER_JOINT:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg/num_heatmaps)

    if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
        heatmaps = [hms[:, :-1] for hms in heatmaps]
        tags = [tms[:, :-1] for tms in tags]

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False
            )
            for tms in tags
        ]

    return outputs, heatmaps, tags




