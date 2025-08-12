from collections import defaultdict
import numpy as np
from copy import deepcopy
import networkx as nx
import math
import matplotlib.pyplot as plt
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff

MAX_DIST = 30
MAX_STEP = 10
# NOISE = 0.5

def get_loc_features(angle_feat_size=4):
    view_angles = get_view_rel_angles()
    view_angles = np.stack(view_angles, 0)
    view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], angle_feat_size)
    return view_ang_fts

def get_view_rel_angles(baseViewId=0):
    rel_angles = np.zeros((12, 2), dtype=np.float32)

    base_heading = 0
    base_elevation = 0
    elevation = 0
    for ix in range(12):
        if ix == 0:
            heading = 0
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation - base_elevation

    return rel_angles

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

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
    if to_clock:
        heading = 2 * np.pi - heading

    elevation = np.arcsin(dz / xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def heading_from_quaternion(quat: np.array):
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    quat = quaternion_from_coeff(quat)
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

def estimate_cand_pos(pos, ori, ang, dis):
    cand_num = len(ang)
    cand_pos = np.zeros([cand_num, 3])

    ang = np.array(ang)
    dis = np.array(dis)
    ang = (heading_from_quaternion(ori) + ang) % (2 * np.pi)
    cand_pos[:, 0] = pos[0] - dis * np.sin(ang)    # x
    cand_pos[:, 1] = pos[1]                        # y
    cand_pos[:, 2] = pos[2] - dis * np.cos(ang)    # z
    return cand_pos


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y and x !=k and y != k:
                    t_dis = self._dis[x][y] + self._dis[y][k]
                    if t_dis < self._dis[x][k]:
                        self._dis[x][k] = t_dis
                        self._dis[k][x] = t_dis
                        self._point[x][k] = y
                        self._point[k][x] = y

        for x in self._dis:
            for y in self._dis:
                if x != y:
                    t_dis = self._dis[x][k] + self._dis[k][y]
                    if t_dis < self._dis[x][y]:
                        self._dis[x][y] = t_dis
                        self._dis[y][x] = t_dis
                        self._point[x][y] = k
                        self._point[y][x] = k

        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self):

        self.graph_nx = nx.Graph()

        self.node_pos = {}          # viewpoint to position (x, y, z)
        self.node_rgb_embeds = {}    # viewpoint to pano feature
        self.node_depth_embeds = {}
        self.node_stepId = {}
        self.node_loc = {}

        self.shortest_path = None
        self.shortest_dist = None
        
        self.node_stop_scores = {}  # viewpoint to stop_score

    def _localize(self, qpos, kpos_dict, loc_noise, ignore_height=False):
        min_dis = 10000
        min_vp = None
        for kvp, kpos in kpos_dict.items():
            if ignore_height:
                dis = ((qpos[[0,2]] - kpos[[0,2]])**2).sum()**0.5
            else:
                dis = ((qpos - kpos)**2).sum()**0.5
            if dis < min_dis:
                min_dis = dis
                min_vp = kvp
        min_vp = None if min_dis > loc_noise else min_vp
        return min_vp
    
    def identify_node(self):
        # assume no repeated node
        # since action is restricted to ghosts
        cur_vp = str(len(self.node_pos))
        return cur_vp


    def update_graph(self, prev_vp, step_id,
                           cur_vp, cur_pos, cur_rgb_embeds, cur_depth_embeds):
        # 1. connect prev_vp
        self.graph_nx.add_node(cur_vp)
        if prev_vp is not None:
            prev_pos = self.node_pos[prev_vp]
            dis = calc_position_distance(prev_pos, cur_pos)
            self.graph_nx.add_edge(prev_vp, cur_vp, weight=dis)

        # 2. update node
        self.node_pos[cur_vp] = cur_pos
        self.node_rgb_embeds[cur_vp] = cur_rgb_embeds
        self.node_depth_embeds[cur_vp] = cur_depth_embeds
        self.node_stepId[cur_vp] = step_id
        self.node_loc[cur_vp] = get_loc_features()

        self.shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph_nx))
        self.shortest_dist = dict(nx.all_pairs_dijkstra_path_length(self.graph_nx))

    def front_to_ghost_dist(self, ghost_vp):
        # assume the nearest front
        min_dis = 10000
        min_front = None
        for front_vp in self.ghost_fronts[ghost_vp]:
            dis = calc_position_distance(
                self.node_pos[front_vp], self.ghost_aug_pos[ghost_vp]
            )
            if dis < min_dis:
                min_dis = dis
                min_front = front_vp
        return min_dis, min_front

    def get_node_rgb_embeds(self, vp):
        return self.node_rgb_embeds[vp]

    def get_node_depth_embeds(self, vp):
        return self.node_depth_embeds[vp]
    

    def get_pos_fts(self, cur_vp, cur_pos, cur_ori, gmap_vp_ids):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vp_ids:
            if cur_vp == vp or vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            # for node
            else:
                base_heading = heading_from_quaternion(cur_ori)
                base_elevation = 0
                vp_pos = self.node_pos[vp]
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    cur_pos, vp_pos, base_heading, base_elevation
                )
                rel_angles.append([rel_heading, rel_elevation])
                shortest_dist = self.shortest_dist[cur_vp][vp]
                shortest_step = len(self.shortest_path[cur_vp][vp])
                rel_dists.append(
                    [rel_dist / MAX_DIST, 
                    shortest_dist / MAX_DIST, 
                    shortest_step / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size=4)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
    
    def locate(self, cpos, kpos, loc_noise):
        gvp = self._localize(cpos, kpos, loc_noise)
        return gvp