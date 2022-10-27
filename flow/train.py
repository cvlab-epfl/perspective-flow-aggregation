from __future__ import print_function, division
import sys
sys.path.append('core')

from core.datasets import fetch_dataloader, render_objects_pytorch3d, GetFlowFromPoseAndDepth
from core.utils import flow_viz

import argparse
import os
import cv2
import time
import random

import yaml

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
from scheduler import WarmupScheduler

import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled=None):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()

def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    model.cuda()
    model.train()

    train_loader = fetch_dataloader(args, workersNum=8)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0

    # pretrained
    if args.restore_ckpt is not None and os.path.exists(args.restore_ckpt):
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print('Pretrained model loaded from %s' % (args.restore_ckpt))
    # checkpoints
    preload_file_name = 'checkpoints/latest.pth'
    if os.path.exists(preload_file_name):
        chkpt = torch.load(preload_file_name, map_location='cpu')  # load checkpoint
        total_steps = chkpt['steps']
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optim'])
        scheduler.load_state_dict(chkpt['sched'])
        print('Weights, optimzer, scheduler are loaded from %s, starting from step %d' % (preload_file_name, total_steps))

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    should_keep_training = True
    while should_keep_training:

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)

        for i_batch, data_blob in pbar:
            if total_steps >= args.num_steps:
                should_keep_training = False
                torch.save(model.state_dict(), 'checkpoints/final.pth')
                print('Training finished')
                break
            total_steps += 1

            optimizer.zero_grad()

            cls_ids, K1, pose1, pose2, image2 = [x.cuda() for x in data_blob]

            # render online
            meshes = [train_loader.dataset.meshes[x][1] for x in cls_ids]
            meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
            image1, depth1 = render_objects_pytorch3d(
                meshes, pose1, K1, 
                train_loader.dataset.patch_width, train_loader.dataset.patch_height
            )
            image1 = image1.permute(0, 3, 1, 2)
            
            # get GT flow
            flow = GetFlowFromPoseAndDepth(pose1, pose2, K1, depth1)

            valid = (flow[:, 0].abs() < 1000) & (flow[:, 1].abs() < 1000)
            flow_predictions = model(image1, image2, iters=args.iters)   

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # logger.push(metrics)
            current_lr = optimizer.param_groups[0]['lr']
            pbar_str = (("steps: %d/%d, lr:%.6f, loss:%.4f") % (total_steps, args.num_steps, current_lr, float(loss)))
            pbar.set_description(pbar_str)

            if total_steps % VAL_FREQ == 0:
                torch.save({
                    'steps': total_steps, 
                    'model': model.state_dict(), 
                    'optim': optimizer.state_dict(),
                    'sched': scheduler.state_dict()
                    },
                    'checkpoints/latest.pth',
                )
                results = {}
                for val_dataset in args.validation:
                    assert (val_dataset == 'occlinemod')
                    results.update(evaluate.validate_bop_flow(model.module, args.render_K, image_list_file=args.validation_file_list, mesh_path=args.mesh_path))

                logger.write_dict(results)
                
                model.train()
            
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument('--name', default='raft-occlinemod', help="name your experiment")
    parser.add_argument('--stage',default='occlinemod', help="determines which dataset to use for training") 
    parser.add_argument('--validation', default=['occlinemod'], type=str, nargs='+')

    parser.add_argument('--training_file_list', default='../data/linemod_hfs/linemod_train.txt')
    parser.add_argument('--validation_file_list', default='../data/linemod_hfs/linemod_test.txt')
    parser.add_argument('--mesh_path', default='../data/linemod_hfs/models/')

    parser.add_argument('--restore_ckpt', default='', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    render_K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ])
    args.render_K = render_K

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
