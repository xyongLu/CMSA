import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import cv2


def setup_logger(final_output_dir, file_name, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    # log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    log_file = '{}_{}.log'.format(file_name, phase)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(format=head)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, time_str


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists() and cfg.RANK == 0:
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    else:
        while not root_output_dir.exists():
            print('=> wait for {} created'.format(root_output_dir))
            time.sleep(30)

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    # final_output_dir = root_output_dir / dataset / model / cfg_name
    final_file_name = dataset +'_'+ model +'_'+ cfg_name
    # final_output_dir = root_output_dir / final_file_name

    # if cfg.RANK == 0:
    #     print('=> creating {}'.format(final_output_dir))
    #     final_output_dir.mkdir(parents=True, exist_ok=True)
    # else:
    #     while not final_output_dir.exists():
    #         print('=> wait for {} created'.format(final_output_dir))
    #         time.sleep(5)

    # logger, time_str = setup_logger(final_output_dir, cfg.RANK, phase)
    logger, time_str = setup_logger(Path(cfg.LOG_DIR), final_file_name, phase)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # return logger, str(final_output_dir), str(tensorboard_log_dir)
    return logger, root_output_dir, Path(cfg.LOG_DIR),final_file_name


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))

    # if is_best and 'state_dict' in states:
    #     torch.save(
    #         states['best_state_dict'],
    #         os.path.join(output_dir, 'model_best.pth.tar')
    #     )


def get_model_summary(model, *input_tensors, item_length=26, verbose=True):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,}".format(flops_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0



## for distribution 
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



def get_multi_scale_size(image, input_size, current_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63)//64 * 64)
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size/w*h+63)//64*64)*current_scale/min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size/h*w+63)//64*64)*current_scale/min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale
    )
    trans = get_affine_transform(center, scale, 0, size_resized)

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
        # (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale



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




#zipreader
import zipfile
import xml.etree.ElementTree as ET

_im_zfile = []
_xml_path_zip = []
_xml_zfile = []
def imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_img = path[pos_at + 1:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def xmlread(filename):
    global _xml_path_zip
    global _xml_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    path_xml = path[pos_at + 2:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found"%(path_zip))
        assert 0
    for i in range(len(_xml_path_zip)):
        if _xml_path_zip[i] == path_zip:
            data = _xml_zfile[i].open(path_xml)
            return ET.fromstring(data.read())
    _xml_path_zip.append(path_zip)
    print("read new xml file '%s'"%(path_zip))
    _xml_zfile.append(zipfile.ZipFile(path_zip, 'r'))
    data = _xml_zfile[-1].open(path_xml)
    return ET.fromstring(data.read())


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

