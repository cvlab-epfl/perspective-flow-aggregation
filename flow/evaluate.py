import sys

from numpy.lib.shape_base import dstack
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.utils.data as data

from core import datasets
from core.utils import frame_utils, flow_viz
from core.utils.utils import InputPadder, forward_interpolate
from core.raft import RAFT

import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer

@torch.no_grad()
def validate_bop_flow(model, render_K, image_list_file='', mesh_path='', iters=12):
    """ Perform evaluation on the bop dataset (test) split """
    model.eval()

    val_dataset = datasets.FlowDatasetFromBOP(render_K, image_list_file=image_list_file,
        mesh_path=mesh_path)
    loader = data.DataLoader(val_dataset, batch_size=8, pin_memory=False, shuffle=False, num_workers=8)
    
    epe_list = []
    pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    for i_batch, data_blob in pbar:
        cls_ids, K1, pose1, pose2, image2 = [x.cuda() for x in data_blob]

        meshes = [val_dataset.meshes[x][1] for x in cls_ids]
        meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
        image1, depth1 = datasets.render_objects_pytorch3d(
            meshes, pose1, K1, 
            val_dataset.patch_width, val_dataset.patch_height
        )
        image1 = image1.permute(0, 3, 1, 2)
        flow_gt = datasets.GetFlowFromPoseAndDepth(pose1, pose2, K1, depth1)
        valid_gt = (flow_gt[:, 0].abs() < 1000) & (flow_gt[:, 1].abs() < 1000)
   
        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=1).sqrt()
        epe_v = epe[valid_gt>=0.5]
        if len(epe_v) > 0:
            epe_list.append(epe_v.mean().item())

        pbar_str = (("validation EPE: %.3f") % (epe_v.mean().item()))
        pbar.set_description(pbar_str)

    epe = np.array(epe_list).mean()
    print("%s EPE: %f" % (image_list_file, epe))
    return {'bop_'+image_list_file: epe}
