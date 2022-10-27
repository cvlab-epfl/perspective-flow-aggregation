

import json
import numpy as np

import os
import cv2

import sys
sys.path.append('flow/core')

import raft
from utils import frame_utils, utils, flow_viz
import datasets

from pose_utils import *

import copy
import torch
import argparse

from tqdm import tqdm

import pytorch3d 

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = torch.device("cpu")

def RefinePoseFromFlows(exemplar_K, target_K, trans_M, patch_vertmaps, patch_forward_flows):
    pseudo_view_count, patch_height, patch_width, _ = patch_vertmaps.shape
    # 
    xy3ds = []
    xy2ds = []
    for i in range(pseudo_view_count):
        valid_flag = (np.linalg.norm(patch_vertmaps[i], axis=2) > 0) \
            & (np.abs(patch_forward_flows[i][..., 0]) < 1000) \
            & (np.abs(patch_forward_flows[i][..., 1]) < 1000)
        
        # add candidates from forward flow
        y1, x1 = valid_flag.nonzero()
        x2 = x1 + patch_forward_flows[i][y1,x1,0]
        y2 = y1 + patch_forward_flows[i][y1,x1,1]
        xy2 = np.concatenate((x2.reshape(1,-1),y2.reshape(1,-1),np.ones_like(x2).reshape(1,-1)), axis=0)
        xy2 = np.matmul(np.linalg.inv(trans_M), xy2) # restore to the raw resolution
        xy2 = np.matmul(np.matmul(target_K, np.linalg.inv(exemplar_K)), xy2) # restore to the raw K
        
        xy3ds.append(patch_vertmaps[i][y1,x1])
        xy2ds.append(xy2[:2].T)

    xy3ds = np.concatenate(xy3ds)
    xy2ds = np.concatenate(xy2ds)
    # 
    ptCnt = len(xy3ds)
    max_ptCnt = 1000
    if ptCnt > max_ptCnt:
        sIdx = np.random.choice(np.arange(0,ptCnt), max_ptCnt, replace=False)
        xy3ds = xy3ds[sIdx]
        xy2ds = xy2ds[sIdx]
    # 
    # newK = np.matmul(trans_M, exemplar_K)
    newK = target_K
    if len(xy2ds) >= 6:
        retval, rot, trans, inliers = cv2.solvePnPRansac(
            xy3ds, xy2ds, newK, None, 
            flags=cv2.SOLVEPNP_EPNP, 
            reprojectionError=3.0, 
            iterationsCount=100
        )
        if retval:
            # print('%d/%d' % (len(inliers), len(xy2ds)))
            R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
            T = trans.reshape(-1, 1)
            return np.concatenate((R, T), axis=1)
        else:
            return None
    return None

def FlowEstimation(flowmodel, image1, image2):
    with torch.no_grad():
        img1s = image1.permute(0, 3, 1, 2)
        img2s = image2.permute(0, 3, 1, 2)

        padder = utils.InputPadder(img1s.shape)
        img1s, img2s = padder.pad(img1s, img2s)

        flow_low, flow_up = flowmodel(img1s, img2s, iters=12, test_mode=True)
        return flow_up

def PoseRefine(predictions, flowModelFile, class_number, meshes, mesh_diameters, render_K):
    classNum = class_number - 1 # get rid of the background class
    new_predictions = copy.deepcopy(predictions)
    #
    # setup flow model
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    args = argparse.Namespace()
    args.model = flowModelFile
    args.small = False
    args.mixed_precision = False
    chkpt = torch.load(args.model)
    flowModel = torch.nn.DataParallel(raft.RAFT(args))
    print("Loading flow model from \"%s\" ..." % flowModelFile)
    if 'model' in chkpt:
        flowModel.load_state_dict(chkpt['model'])
    else:
        flowModel.load_state_dict(chkpt)
    flowModel = flowModel.module
    flowModel.to(device)
    flowModel.eval()
    #
    patch_width = 256
    patch_height = 256
    #
    pbar = tqdm(enumerate(predictions.items()), total=len(predictions), dynamic_ncols=True)
    for _, data_blob in pbar:
        filename, item = data_blob

        pbar.set_description(filename)

        iImg = cv2.imread(filename)
        #
        K = np.array(item['meta']['K'])
        width = item['meta']['width']
        height = item['meta']['height']
        new_predictions[filename]['pred'] = [] # clear new predictions
        for score, clsid, predR, predT in item['pred']:
            # find the correspoding ground truth
            if clsid not in item['meta']['class_ids']:
                continue
            locIdx = item['meta']['class_ids'].index(clsid) # assert having one and only one
            gtR = np.array(item['meta']['rotations'][locIdx])
            gtT = np.array(item['meta']['translations'][locIdx])
            gtP = np.concatenate((gtR, gtT), axis=1)
            #
            predR = np.array(predR)
            predT = np.array(predT)
            predP = np.concatenate((predR, predT), axis=1)
            if False in np.isfinite(predR).reshape(-1).tolist():
                continue
            if False in np.isfinite(predT).reshape(-1).tolist():
                continue

            # get the patch cropping matrix
            bbox_reproj = np.matmul(render_K, np.matmul(predR, meshes[clsid][0].bounding_box_oriented.vertices.T) + predT)
            xs = bbox_reproj[0] / bbox_reproj[2]
            ys = bbox_reproj[1] / bbox_reproj[2]
            trans_M = datasets.get_crop_M(xs, ys, patch_width, patch_height)

            # render the patch
            image1, depth1 = datasets.render_objects_pytorch3d(
                pytorch3d.structures.join_meshes_as_batch([meshes[clsid][1]]),
                torch.from_numpy(predP).unsqueeze(0).float(), 
                torch.from_numpy(np.matmul(trans_M, render_K)).unsqueeze(0).float(), 
                patch_width, patch_height
            )

            # crop 
            trans_K = np.matmul(render_K, np.linalg.inv(K)) # align the input K and the K of exemplars
            image2 = cv2.warpAffine(iImg[:,:,:3], np.matmul(trans_M, trans_K)[:2], (patch_width, patch_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = torch.from_numpy(image2).unsqueeze(0).float().to(image1.device)

            # get 3D points
            vert_map, _ = datasets.GetVertmapFromDepth(
                torch.from_numpy(np.matmul(trans_M, render_K)).unsqueeze(0).float().to(depth1.device), 
                torch.from_numpy(predP).unsqueeze(0).float().to(depth1.device), 
                depth1
            )
            vert_map = vert_map.permute(0,2,3,1).cpu().numpy()

            # estimating 2D-to-2D correspondence
            pred_flows1 = FlowEstimation(flowModel, image1, image2)
            pred_flows1 = pred_flows1.permute(0,2,3,1).cpu().numpy()

            # relay to 3D-to-2D and refine
            refP = RefinePoseFromFlows(render_K, K, trans_M, vert_map, pred_flows1)

            if refP is None:
                # refinement is failed, save the raw prediction
                new_predictions[filename]['pred'].append([score, clsid, predR, predT])
            else:
                refR = refP[:,:3]
                refT = refP[:,3].reshape(-1,1)
                new_predictions[filename]['pred'].append([score, clsid, refR, refT])

    return new_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_pose_file', default='./wdr_init.json', type=str)
    parser.add_argument('--mesh_dir', default='./data/linemod_hfs/models/', type=str)
    parser.add_argument('--flow_model_file', default='./linemod.pth', type=str)

    args = parser.parse_args()
    
    n_class = 14
    mesh_diameters = [104.26,250.85,177.43,204.83,154.63,264.12,110.83,164.65,178.35,145.61,279.04,287.24,213.25]
    symmetry_types = {
        "cls_7": 1, 
        "cls_8": 1,
    }
    render_K = np.array([
        [572.4114, 0.0, 325.2611],
        [0.0, 573.57043, 242.04899],
        [0.0, 0.0, 1.0]
    ])

    json_file_name = args.init_pose_file
    flowModelFile = args.flow_model_file

    # load initial poses
    with open(json_file_name, 'r') as f:
        preds = json.load(f)
        print("Loading initial poses from \"%s\" ..." % json_file_name)

    # load meshes
    meshes, objID_2_clsID = datasets.load_bop_meshes(args.mesh_dir, DEVICE)

    # evaluate before
    print("Before PFA refinement:")
    accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate_pose_predictions(preds, n_class, meshes, mesh_diameters, symmetry_types)
    print_accuracy_per_class(accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class)

    # refinement
    new_preds = PoseRefine(preds, flowModelFile, n_class, meshes, mesh_diameters, render_K)

    # evaluate after
    print("After PFA refinement:")
    accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class, accuracy_adi_per_depth, accuracy_rep_per_depth, depth_range \
        = evaluate_pose_predictions(new_preds, n_class, meshes, mesh_diameters, symmetry_types)
    print_accuracy_per_class(accuracy_adi_per_class, accuracy_auc_per_class, accuracy_rep_per_class)
