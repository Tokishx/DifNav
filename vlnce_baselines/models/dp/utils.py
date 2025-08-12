import math
from turtle import heading
import torch
import numpy as np


def angle_feature(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 64, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
                [math.sin(head), math.cos(head)] * (64 // 2))

    return heading_enc.to(device)

def dir_angle_feature(angle_list, device=None):
    feature_dim = 64
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
                [math.sin(angle_rad), 
                math.cos(angle_rad)] * (feature_dim // 2))

    return heading_enc


def angle_feature_with_ele(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 128, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
            [
                math.sin(head), math.cos(head),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc.to(device)

def angle_feature_torch(headings: torch.Tensor):
    return torch.stack(
        [
            torch.sin(headings),
            torch.cos(headings),
            torch.sin(torch.zeros_like(headings)),
            torch.cos(torch.zeros_like(headings))
        ]
    ).float().T

def dir_angle_feature_with_ele(angle_list, device=None):
    feature_dim = 128
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
            [
                math.sin(angle_rad), math.cos(angle_rad),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc
    

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0, to_clock=False):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    # heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
    # if b[1] < a[1]:
    #     heading = np.pi - heading
    if b[2] > a[2]:
        heading = np.pi - heading
    heading -= base_heading
    # 逆时针
    if to_clock:
        heading = 2 * np.pi - heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist



def get_cur_angle(path, start_heading):
    '''
    假设所有图片的绝对方向都是一致的，只需要调整agent的朝向
    '''
    # trajectory < 2, heading = start_heading
    if len(path) < 2:
        heading = start_heading
        elevation = 0
    
    # trajectory > 2 heading = prev_vp tp cur_vp's rotation
    # 根据habitat逻辑，需要先转向再直行，即不发生collsion的情况下，坐标之间的相对朝向就是cur_rotation
    else:
        prev_vp = path[-2]
        cur_vp = path[-1]
        dx = prev_vp[0] - cur_vp[0]
        dy = prev_vp[1] - cur_vp[1]
        dz = prev_vp[2] - cur_vp[2]
        # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
        # habitat z == y
        xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
        xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)
        heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
        elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    return heading, elevation


def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts
