
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import cv2
import transforms3d
import time

import trimesh
import psutil
from scipy.spatial import transform
import copy

import os
import math
import json
import random
from glob import glob
import os.path as osp

from utils import frame_utils, utils, flow_viz, pose_viz

import pytorch3d
import pytorch3d.structures
import pytorch3d.renderer

def load_bop_meshes(model_path, device='cuda'):
    # load meshes
    meshFiles = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    meshFiles.sort()
    meshes = []
    scaling_factors = []
    objID_2_clsID = {}
    for i in range(len(meshFiles)):
        mFile = meshFiles[i]
        objId = int(os.path.splitext(mFile)[0][4:])
        objID_2_clsID[str(objId)] = i
        # 
        tMesh = trimesh.load(model_path + mFile)
        # Pytorch3D APIs
        verts = torch.from_numpy(tMesh.vertices)[None].float().to(device)
        faces = torch.from_numpy(tMesh.faces)[None].float().to(device)
        features = torch.from_numpy(tMesh.visual.vertex_colors[:,:3] / 255.0)[None].float()
        tex = pytorch3d.renderer.TexturesVertex(verts_features=features).to(device)
        p3d_mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, textures=tex)
        meshes.append([tMesh, p3d_mesh])
        # 
        # print('mesh from "%s" is loaded' % (model_path + mFile))
    # 
    return meshes, objID_2_clsID

def render_objects_pytorch3d(meshes, poses, K, w, h):
    scaling_factor = np.power(10, -np.round(np.log10(torch.norm(meshes._verts_list[0], dim=1).mean().tolist())))
    meshes = meshes.scale_verts(scaling_factor)
    # 
    device = meshes.device
    batch_size = len(poses)
    assert(meshes._N == batch_size)
    # 
    focal_length = torch.cat((K[:,0,0].view(-1,1), K[:,1,1].view(-1,1)), axis=1)
    principal_point = torch.cat((K[:,0,2].view(-1,1), K[:,1,2].view(-1,1)), axis=1)
    image_size = (h, w)
    image_size = np.array(image_size).reshape(1,2).repeat(batch_size,axis=0).astype(np.int)
    # 
    Rs = []
    Ts = []
    deltaR = torch.eye(3).to(poses.device)
    deltaR[0][0] = -1
    deltaR[1][1] = -1
    for p in poses:
        R = torch.matmul(deltaR, p[:3,:3]).T
        T = p[:3, 3] * scaling_factor
        T[0] = -T[0]
        T[1] = -T[1]
        Rs.append(R)
        Ts.append(T)
    Rs = torch.stack(Rs)
    Ts = torch.stack(Ts)
    # 
    cameras = pytorch3d.renderer.PerspectiveCameras(device=device, \
        focal_length=focal_length, principal_point=principal_point, \
        in_ndc=False, image_size=image_size, R=Rs, T=Ts)
    raster_settings = pytorch3d.renderer.RasterizationSettings(image_size=(h, w), \
        blur_radius=0, faces_per_pixel=1, max_faces_per_bin = meshes._num_faces_per_mesh.max(), \
        perspective_correct=True)
    rasterizer = pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # 
    lights = pytorch3d.renderer.AmbientLights(device=device)
    shader = pytorch3d.renderer.SoftPhongShader(device=device, cameras=cameras, lights=lights)
    # 
    fragments = rasterizer(meshes)
    images = shader(fragments, meshes) * 255
    #
    depths = (fragments.zbuf / scaling_factor)
    depths[depths < 0] = 0 # the default depth is negative, set invalid to 0
    # 
    return images[...,:3], depths.squeeze(-1)

def fetch_dataloader(args, workersNum = 8):
    """ Create the data loader for the corresponding trainign set """

    assert (args.stage == 'occlinemod')
    train_dataset = FlowDatasetFromBOP(args.render_K, image_list_file=args.training_file_list, mesh_path=args.mesh_path)

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, 
        num_workers=workersNum,  
        drop_last=True
    )

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

def is_memory_enough():
    mem_stat = psutil.virtual_memory()
    if mem_stat.percent < 90 and mem_stat.available > 1024*1024*1024: # leave 1G at least
        return True
    return False

def load_image_cached(img_path, mem_cache=None):
    if mem_cache != None and not img_path in mem_cache:
        if is_memory_enough():
            with open(img_path, 'rb') as f:
                mem_cache[img_path] = f.read()
    if mem_cache != None and img_path in mem_cache:
        return cv2.imdecode(np.fromstring(mem_cache[img_path], np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

def load_json_cached(json_path, mem_cache=None):
    if mem_cache != None and not json_path in mem_cache:
        if is_memory_enough():
            with open(json_path, 'r') as f:
                mem_cache[json_path] = f.read()
    if mem_cache != None and json_path in mem_cache:
        return json.loads(mem_cache[json_path])
    else:
        return json.load(open(json_path))
        
def get_single_bop_annotation(img_path, objID_2_clsID, mem_cache=None):
    img_path = img_path.strip()
    # 
    gt_dir, tmp, imgName = img_path.rsplit('/', 2)
    assert(tmp == 'rgb')
    imgBaseName, _ = os.path.splitext(imgName)
    # 
    camera_file = gt_dir + '/scene_camera.json'
    gt_file = gt_dir + "/scene_gt.json"
    # gt_info_file = gt_dir + "/scene_gt_info.json"
    gt_mask_visib = gt_dir + "/mask_visib/"

    gt_json = load_json_cached(gt_file, mem_cache)
    cam_json = load_json_cached(camera_file, mem_cache)
    # gt_info_json = json.load(open(gt_info_file))

    try:
        im_id_str = str(int(imgBaseName))
    except:
        im_id_str = "UNDEFINED"
    #
    if im_id_str in cam_json:
        annot_camera = cam_json[im_id_str]
    else:
        annot_camera = cam_json[imgBaseName]
    # 
    if im_id_str in gt_json:
        annot_poses = gt_json[im_id_str]
    else:
        annot_poses = gt_json[imgBaseName]
    # 
    objCnt = len(annot_poses)
    K = np.array(annot_camera['cam_K']).reshape(3,3)

    class_ids = []
    # bbox_objs = []
    rotations = []
    translations = []
    merged_mask = None
    instance_idx = 1
    for i in range(objCnt):
        mask_vis_file = gt_mask_visib + ("%s_%06d.png" %(imgBaseName, i))
        mask_vis = load_image_cached(mask_vis_file, mem_cache)
        height = mask_vis.shape[0]
        width = mask_vis.shape[1]
        if merged_mask is None:
            merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
        # 
        R = np.array(annot_poses[i]['cam_R_m2c']).reshape(3,3)
        T = np.array(annot_poses[i]['cam_t_m2c']).reshape(3,1)
        obj_id = str(annot_poses[i]['obj_id'])
        if not obj_id in objID_2_clsID:
            continue
        cls_id = objID_2_clsID[obj_id]
        # 
        # bbox_objs.append(bbox)
        class_ids.append(cls_id)
        rotations.append(R)
        translations.append(T)
        # compose segmentation labels
        merged_mask[mask_vis==255] = instance_idx
        instance_idx += 1
    
    return K, merged_mask, class_ids, rotations, translations


def GetVertmapFromDepth(K, pose1, depth1):
    batch_size, height, width = depth1.shape

    # get vert map from depth1
    xs = torch.linspace(0, width-1, steps=width, device=depth1.device)
    ys = torch.linspace(0, height-1, steps=height, device=depth1.device)
    y, x = torch.meshgrid(ys, xs)
    xy1 = torch.stack([x, y, torch.ones_like(x)]).reshape(3,-1)
    xy1 = xy1.repeat(batch_size, 1, 1)
    xyn1 = torch.matmul(torch.inverse(K), xy1)

    # get 3D points in Camera coordinates
    xyzc = xyn1 * depth1.reshape(batch_size, 1, -1)
    # change to object coordinates
    vert_map = torch.matmul(pose1[...,:3].transpose(1,2), xyzc-pose1[...,3].unsqueeze(-1))
    vert_map = vert_map.reshape(batch_size, 3, height, width)

    valid_flag = (depth1 < 1e-6).unsqueeze(1).repeat(1,3,1,1)
    vert_map[valid_flag] = 0

    return vert_map, xy1

def GetFlowFromPoseAndDepth(pose1, pose2, K, depth1):
    batch_size, height, width = depth1.shape

    # get vertex map in object frame
    vert_map, xy1 = GetVertmapFromDepth(K, pose1, depth1)

    # compute new reprojections under the second pose
    R2 = pose2[..., :3]
    T2 = pose2[..., 3].unsqueeze(-1)
    xyp2 = torch.matmul(K, torch.matmul(R2, vert_map.reshape(batch_size, 3, -1)) + T2)
    x2 = xyp2[:,0] / xyp2[:,2]
    y2 = xyp2[:,1] / xyp2[:,2]

    # value to use to represent unknown flow
    UNKNOWN_FLOW = 1e10
    flowx = x2 - xy1[:,0]
    flowy = y2 - xy1[:,1]
    invalid_mask = (depth1.reshape(batch_size, -1) < 1e-6)
    flowx[invalid_mask] = UNKNOWN_FLOW
    flowy[invalid_mask] = UNKNOWN_FLOW
    flow = torch.stack([flowx, flowy]).permute(1,0,2).reshape(batch_size, 2, height, width)
    return flow

def get_crop_M(mask_xs, mask_ys, patch_width, patch_height):
    patch_border = 5
    if (len(mask_ys) < 3):
        return None
    raw_patch_cx = (mask_xs.max() + mask_xs.min()) / 2
    raw_patch_cy = (mask_ys.max() + mask_ys.min()) / 2
    raw_patch_w = mask_xs.max() - mask_xs.min() + 2*patch_border
    raw_patch_h = mask_ys.max() - mask_ys.min() + 2*patch_border
    scale = min(patch_width/raw_patch_w, patch_height/raw_patch_h)
    pleft = patch_width/2 - raw_patch_cx*scale
    ptop = patch_height/2 - raw_patch_cy*scale
    trans_M = np.array([[scale, 0.0, pleft], [0.0, scale, ptop], [0.0, 0.0, 1.0]])  # transformation matrix
    return trans_M

def AddJitter2Pose(K, R, T, max_angle=10, max_pix_offset=10):
    # generate random delta R
    angle_th = max_angle * np.pi / 180 # degree
    direct = np.random.rand(3)
    direct /= np.linalg.norm(direct)
    angle = np.random.uniform(-angle_th, angle_th)
    deltaR = transform.Rotation.from_rotvec(angle*direct).as_matrix()
    newR = np.matmul(deltaR, R)
    #
    # generate random delta T
    raw_z = T.reshape(-1)[2]
    origin_pos = np.matmul(K, T)
    ox = float(origin_pos[0]/origin_pos[2])
    oy = float(origin_pos[1]/origin_pos[2])
    corner_uvs = np.array([
        [ox - max_pix_offset, oy - max_pix_offset, 1], 
        [ox - max_pix_offset, oy + max_pix_offset, 1], 
        [ox + max_pix_offset, oy - max_pix_offset, 1], 
        [ox + max_pix_offset, oy + max_pix_offset, 1]
        ]).T
    corner_xyz = np.matmul(np.linalg.inv(K), corner_uvs * raw_z)
    x_min = corner_xyz[0].min()
    x_max = corner_xyz[0].max()
    y_min = corner_xyz[1].min()
    y_max = corner_xyz[1].max()
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = random.uniform(0.9, 1.1) * raw_z
    newT = np.array([x,y,z]).reshape(-1,1)
    return newR, newT

# memory cache, a shared global dict (company with multiple workers in PyTorch)
import multiprocessing
g_mem_cache = multiprocessing.Manager().dict()
# g_mem_cache = None

class FlowDatasetFromBOP(data.Dataset):
    def __init__(self, render_K, image_list_file='', mesh_path=''):
        self.render_K = render_K
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        # 
        self.patch_width = 256
        self.patch_height = 256
        # file list and data are typically in the same directory
        dataDir = os.path.split(image_list_file)[0]
        with open(image_list_file, 'r') as f:
            tmp_img_files = f.readlines()
            for i in range(len(tmp_img_files)):
                tmp_img_files[i] = dataDir + os.sep + tmp_img_files[i].strip()
        self.meshes, self.objID_2_clsID = load_bop_meshes(mesh_path)
        self.image_list = tmp_img_files

    def __getitem__(self, index):
        # read the pose dataset
        while True:
            pose_image_path = self.image_list[index]
            K, merged_mask, class_ids, rotations, translations = get_single_bop_annotation(pose_image_path, self.objID_2_clsID, mem_cache=g_mem_cache)
            # only randomly pick up one instance
            instanceIdx = random.randint(0,len(class_ids)-1)
            current_mask = (merged_mask==(instanceIdx+1))
            if current_mask.sum() < 100:
                index = random.randint(0,len(self.image_list)-1)
                continue
            input_cvImg = cv2.imread(pose_image_path)
            height, width, _ = input_cvImg.shape
            # add some noise to the query R and T
            gtR = rotations[instanceIdx]
            gtT = translations[instanceIdx]
            #
            #queryR, queryT = gtR, gtT
            queryR, queryT = AddJitter2Pose(K, gtR, gtT)
            break

        # get the patch cropping matrix
        bbox_reproj = np.matmul(self.render_K, np.matmul(queryR, self.meshes[class_ids[instanceIdx]][0].bounding_box_oriented.vertices.T) + queryT)
        xs = bbox_reproj[0] / bbox_reproj[2]
        ys = bbox_reproj[1] / bbox_reproj[2]
        trans_M = get_crop_M(xs, ys, self.patch_width, self.patch_height)
        
        # crop the input image
        trans_K = np.matmul(self.render_K, np.linalg.inv(K)) # align the input K and the K of exemplars
        patch_input = cv2.warpAffine(input_cvImg[:,:,:3], np.matmul(trans_M, trans_K)[:2], (self.patch_width, self.patch_height), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))

        img2 = cv2.cvtColor(patch_input, cv2.COLOR_BGR2RGB)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return [
            class_ids[instanceIdx], 
            torch.from_numpy(np.matmul(trans_M, self.render_K)).float(), 
            torch.from_numpy(np.concatenate((queryR, queryT), axis=1)).float(), 
            torch.from_numpy(np.concatenate((gtR, gtT), axis=1)).float(), 
            img2
        ]

    def __rmul__(self, v):
        # self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
