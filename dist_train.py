import sys
import os
import argparse
import time
import datetime
import numpy as np
from pathlib import Path
from ptflops import get_model_complexity_info

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
from timm.utils import NativeScaler,get_state_dict, ModelEma

import lib.dataset  as dataset
from engine import train_one_epoch, evaluate
from losses import JointsMSELoss, CombinedTargetMSELoss, JointsOHKMMSELoss
from config import cfg
from config import update_config
from samplers import RASampler
from mixup import Mixup
from models import cmasformer
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on {}".format(device))


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--uni-note', default='', type=str, help='unique note on the  name of model to train')
    parser.add_argument('--model', type=str,      default='CMSAFormer_L_32_3g')
    parser.add_argument('--cfg',
                        default='experiments/coco/vits/cmsa_l_32.yaml',
                        help='experiment configure file name',
                        # required=True,
                        type=str)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency.')

    parser.add_argument('--epochs', default=300, type=int, help='Maximum number of training epochs.')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--input-size', type=int,      default=32)
    parser.add_argument('--patch-size', type=int,      default=2)
    parser.add_argument('--batch-size', default=48, type=int,  help='Batch size for training.')
    parser.add_argument('--batch-size-test', default=10, type=int,  help='Batch size for testing.')

    parser.add_argument('--drop', type=float, default=0., metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (defaudevice = torch.device(args.device)ult: None, no clipping)')
    parser.add_argument('--clip-grad', type=float, default=5, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 2.5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " +  "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',  help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',  help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,  help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--dataset', default='coco', type=str,  help='Dataset type.')
    parser.add_argument('--data-dir',  default='../../Data/coco2017', type=str, help='Directory path for data.')

    parser.add_argument('--filename-list', type=str, default='data/300W_LP_filename_filtered.txt',  help='Path to text file containing relative paths for every example.')
    parser.add_argument('--filename-list-test', type=str, default='data/AFLW2000_filename_filtered.txt',  help='Path to text file containing relative paths for every example.',)
    parser.add_argument('--alpha', dest='alpha', default=2, type=float, help='Regression loss coefficient.')
    parser.add_argument('--num-bins', type=int, default=66, help='number of bins that bin each angle')
    parser.add_argument('--bin-interval', type=int, default=3, help='bin interval for each classification style. 1--198, 3--66, 11--18, 33--6, 99--2')
    parser.add_argument('--finetune', default=False, help='finetune from checkpoint')
    
    parser.add_argument('--output-dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    # parser.add_argument("--gpus", type=int, default=1, help="the number of GPUs, (default: 1)")
    parser.add_argument('--seed',       type=int,  default=0,           help='random seed for all')
    parser.add_argument('--resume', default= '' , help='resume from checkpoint')
    parser.add_argument('--log-step',   type=int,      default=100)
    parser.add_argument('--log-dir', default='./logs', help='path where to logs, empty for no saving')
    # parser.add_argument('--use-cuda',   type=utils.str2bool, default=True,        help='enables cuda')
    parser.add_argument('--num_workers',  type=int,      default=4)
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args

def show_viz_loss(args, epoch, loss, vis, plot_data):
    #Visdom Visualization
    if (epoch % 2 == 0) and (vis != None) and (plot_data != None):
        plot_data['X'].append(epoch)
        plot_data['Y'].append(
            round(loss,4)
        )
        vis.line(
            #X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            X=np.array(plot_data['X']),
            Y=np.array(plot_data['Y']),
            opts={
                'title': 'MAE over times',
                'legend': plot_data['legend'],
                'xlabel': 'Iterations:' + str(epoch),
                'ylabel': 'MAE',
                'width': 1200,
                'height': 390,
            },
        win = 'MAE evaluated on the {}'.format(args.dataset_test) 
        )

def main(args):
    utils.init_distributed_mode(args)

    update_config(cfg, args)
    print(cfg)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print('Loading data.')
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([ transforms.ToTensor(), 
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = eval('dataset.'+cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,transform_train)
    dataset_val = eval('dataset.'+cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,transform_val)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    print("batchs of examples for training on {}:{}".format(cfg.DATASET.TRAIN_SET, len(data_loader_train)))
    print("batchs of examples for evaluating on {}:{}".format(cfg.DATASET.TEST_SET, len(data_loader_val)))

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=cfg.MODEL.NUM_JOINTS)

    model = create_model(
    args.model,
    pretrained=False,
    # img_size = args.input_size,
    num_classes= cfg.MODEL.NUM_JOINTS,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None)

    print("args.finetune {}".format(args.finetune))
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print("Removing key {} from pretrained checkpoint".format(k))
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).to(device)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            model_without_ddp.load_state_dict(checkpoint)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
    
    print('{} is ready to train.'.format(args.model))
    best_perf = 0.0
    train_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        epoch_start_time = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(cfg, model, data_loader_train, criterion, optimizer, device, epoch)

        lr_scheduler.step(epoch)
        # evaluate on validation set
        if utils.is_main_process():
            perf_indicator = evaluate(cfg, model, data_loader_val, dataset_val, criterion, device, args.output_dir)
        else:
            # perf_indicator = 0.0
            continue

        if perf_indicator >= best_perf:
            print('Taking snapshot...')
            if args.distributed:
                model_state_dict = model.module.state_dict()  # 多GPU
            else:
                model_state_dict = model.state_dict()  # 单GPU

            state = {
                'model': model_state_dict,
                'perf_history':perf_indicator,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                # 'model_ema': get_state_dict(model_ema),
                'args': args,
            }
            filename = "{}_{}_checkpoint_{}".format(args.model, args.dataset, args.uni_note)
            # utils.save_on_master(state, args.output_dir + '/' + filename+'.pth')
            torch.save(state, args.output_dir + '/' + filename+'.pth')
            best_perf = perf_indicator

            print("the better tested error_history on {} : {}".format(args.dataset, state['perf_history']))
            with open(args.output_dir +"/" + filename+".txt", "a+") as f:
                f.write("Traindate: " + str(datetime.datetime.now()) + " Epoch: " + str(epoch) + " " +
                        'AP value: %.4f' % (best_perf) + "\n")
                f.write("\n")
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        print('Learning rate is %s' % [v['lr']
              for v in optimizer.param_groups][0])
        print('Evaluation error in degrees of the model on the ' + args.dataset + ':')
        print('Epoch time:[{}], Best AP:[{:.4f}]'.format(epoch_time_str, best_perf))
        
    total_time = time.time() - train_start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Training Time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()
    main(args)
    