import torch
import torch.distributed as dist
import numpy as np
import math
import copy
import h5py

class ARGS():
    def __init__(self):
        self.local_rank = 0

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            # print(tensor)
            tensor /= world_size

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def repeat_allocation(allocations, max_number):
    if torch.is_tensor(max_number):
        max_number = max_number.long().item()
    else:
        max_number = max_number.long()
    allocation_number = len(allocations)
    repeat_time, res = max_number // allocation_number, max_number % allocation_number
    allocations_ = []
    for i in range(repeat_time):
        allocations_ += copy.deepcopy(allocations)
    allocations_ += copy.deepcopy(allocations)[:res]

    return allocations_


def allocate(number, ep_length, size_per_time):
    length_to_indexes = {ep_length[i]: [] for i in
                        range(len(ep_length))}
    for i in range(len(ep_length)):
        length_to_indexes[ep_length[i]] += [i]*number[i]

    values = []
    for i in range(len(number)):
        values += [ep_length[i]] * number[i]

    groups = int((len(values) - 0.01) // size_per_time + 1)

    values.sort(reverse=True)

    load_balance_groups = [[] for grp in range(groups)]

    for v in values:
        load_balance_groups.sort(key=lambda x: sum(x))
        load_balance_groups[0].append(v)

    indexes = []
    set_length = list(set(ep_length))
    for i in range(groups):
        index = np.zeros(len(load_balance_groups[i]),dtype=int)
        for j in range(len(set_length)):
            length_indexes = length_to_indexes[set_length[j]]
            position = np.where(np.array(load_balance_groups[i]) ==
                          set_length[j])[0]
            position_length = len(position)
            index[position] = length_indexes[:position_length]
            length_to_indexes[set_length[j]] = length_indexes[position_length:]
        indexes.append((index).tolist())

    return indexes

def allocate_instructions(instruction_lengths, allocations,ep_length, instruction_ids):
    instruction_ids_copy = copy.deepcopy(instruction_ids)
    allocations_copy = copy.deepcopy(allocations)
    instruction_lengths_copy = copy.deepcopy(instruction_lengths)
    values = []
    value_indexes = []
    weights = []
    for i in range(len(instruction_lengths)):
        instruction_length = instruction_lengths[i]
        values += instruction_length
        value_indexes += len(instruction_length)*[i]
        weights += [ep_length[i]] * len(instruction_length)
    # values = np.array(values)
    # value_indexes = np.array(value_indexes)
    values = np.array(values)
    weights = np.array(weights)
    value_indexes = np.array(value_indexes)
    sorted_index = np.argsort(values*weights)[::-1]
    values = values[sorted_index]
    value_indexes = value_indexes[sorted_index]
    weights = weights[sorted_index]

    groups = len(allocations)
    load_balance_groups = [[] for grp in range(groups)]
    group_weights = [[] for grp in range(groups)]
    instruction_allocations = [[] for grp in range(groups)]
    for j in range(len(values)):
        summation = np.array([np.sum(np.array(load_balance_groups[i])*np.array(group_weights[i])) for i in range(groups)])
        sorted_index = np.argsort(summation)
        for i in sorted_index:
            index = value_indexes[j]
            value = values[j]
            if index in allocations_copy[i]:
                allocations_copy[i].remove(index)
                load_balance_groups[i].append(value)
                group_weights[i].append(weights[j])
                # check[i].append(index)
                index_in_length = np.where(np.array(instruction_lengths_copy[index]) == value)[0][0]
                instruction_lengths_copy[index].pop(index_in_length)
                instruction_allocations[i].append(instruction_ids_copy[index].pop(index_in_length))
                break

    return instruction_allocations


def allocate_by_scene_for_ddp(number, ep_length, size_per_time):
    length_to_indexes = {ep_length[i]: [] for i in
                        range(len(ep_length))}
    for i in range(len(ep_length)):
        length_to_indexes[ep_length[i]] += [i]*number[i]

    values = []
    for i in range(len(number)):
        values += [ep_length[i]] * number[i]

    groups = int((len(values) - 0.01) // size_per_time + 1)

    values.sort(reverse=True)

    load_balance_groups = [[] for grp in range(groups)]

    for v in values:
        load_balance_groups.sort(key=lambda x: sum(x))
        load_balance_groups[0].append(v)

    indexes = []
    set_length = list(set(ep_length))
    for i in range(groups):
        index = np.zeros(len(load_balance_groups[i]),dtype=int)
        for j in range(len(set_length)):
            length_indexes = length_to_indexes[set_length[j]]
            position = np.where(np.array(load_balance_groups[i]) ==
                          set_length[j])[0]
            position_length = len(position)
            index[position] = length_indexes[:position_length]
            length_to_indexes[set_length[j]] = length_indexes[position_length:]
        indexes.append((index).tolist())

    return indexes


def get_camera_orientations12():
    base_angle_deg = 30
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,12):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict


def get_camera_orientations24():
    base_angle_deg = 15
    base_angle_rad = math.pi / 12
    orient_dict = {}
    for k in range(1,24):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask


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

def get_traj_pano_fts(scan, vp, device):
    '''
    Tokens in each pano: [cand_views, noncand_views, objs]
    Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
    '''

    view_fts, dep_fts = get_scanvp_feature(scan, vp)

    view_img_fts, view_dep_fts = [], []
    view_img_fts = view_fts
    view_dep_fts = dep_fts
    # combine cand views and noncand views
    view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
    view_img_fts = torch.from_numpy(view_img_fts).to(device)
    view_dep_fts = np.stack(view_dep_fts, 0)
    view_dep_fts = torch.from_numpy(view_dep_fts).to(device)
        

    return view_img_fts, view_dep_fts


def get_scanvp_feature(scan, viewpoint):
    img_ft_file = '/workspace/vlnce38/visualnav-transformer/train/vln_features/img_features/CLIP-ViT-B-32-views-habitat.hdf5'
    dep_ft_file = '/workspace/vlnce38/visualnav-transformer/train/vln_features/depth_features/resnet-views-habitat.hdf5'
    key = '%s_%s' % (scan, viewpoint)
    with h5py.File(img_ft_file, 'r') as f:
        view_fts = f[key][...].astype(np.float32)
    with h5py.File(dep_ft_file, 'r') as f:
        dep_fts = f[key][...].astype(np.float32)
    return view_fts, dep_fts


def get_cur_pos_ori(gt_path, stepk, start_heading=None):
    cur_vp = gt_path[-1]
    if start_heading:
        heading = start_heading
        elevation = 0
    else:
        prev_vp = gt_path[-2]
        dx = prev_vp[0] - cur_vp[0]
        dy = prev_vp[1] - cur_vp[1]
        dz = prev_vp[2] - cur_vp[2]
        # xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
        # habitat z == y
        xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
        xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)
        heading = np.arcsin(-dx / xz_dist)  # [-pi/2, pi/2]
        elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    return cur_vp, (heading, elevation)



    