import gc
import os
import sys
import random
import yaml
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines


import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils_dp import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss
from vlnce_baselines.models.dp.utils.utils import calculate_world_coordinates, calculate_world_coordinates_from_polar, calculate_vp_ori

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele, get_traj_pano_fts
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from habitat_extensions.utils import heading_from_quaternion
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks, pad_tensors
from torch.nn.utils.rnn import pad_sequence


@baseline_registry.register_trainer(name="SS-ETP-DP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.net = self.policy.net.module
            else:
                # TODO
                raise ValueError

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")
        return start_iter
        
    def _nav_gmap_variable(self, txt_id, cur_vp, cur_pos, cur_ori):
        batch_txt_id = []
        batch_gmap_step_ids, batch_gmap_lens, batch_gmap_pos_fts = [], [], []
        batch_traj_img_fts, batch_traj_depth_fts, batch_traj_loc_fts = [], [], []

        for i, gmap in enumerate(self.gmaps):
            # instruction
            txt_ids = torch.LongTensor(txt_id[i])

            # traj features
            node_vp_ids = list(gmap.node_pos.keys())
            node_vp_ids = node_vp_ids[-3:]

            traj_img_fts = [gmap.get_node_rgb_embeds(vp) for vp in node_vp_ids]
            traj_depth_fts = [gmap.get_node_depth_embeds(vp) for vp in node_vp_ids]
            traj_loc_fts = [torch.from_numpy(gmap.node_loc[vp]) for vp in node_vp_ids]             

            # gmap features
            gmap_step_ids = [id for id in range(1, len(node_vp_ids)+1)]
            gmap_pos_fts = torch.from_numpy(
                gmap.get_pos_fts(cur_vp[i], cur_pos[i], cur_ori[i], node_vp_ids)
            )
            batch_txt_id.append(txt_ids)
            batch_traj_img_fts.append(traj_img_fts)
            batch_traj_depth_fts.append(traj_depth_fts)
            batch_traj_loc_fts.append(traj_loc_fts)

            batch_gmap_pos_fts.append(gmap_pos_fts)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
     
        # collate
        # instruction
        batch_txt_len = torch.LongTensor([len(x) for x in batch_txt_id]).cuda()
        batch_txt_id = pad_sequence(batch_txt_id, batch_first=True, padding_value=0).cuda()

        # trajectory
        batch_traj_step_lens = [len(x) for x in batch_traj_img_fts]
        batch_traj_vp_view_lens = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch_traj_img_fts], [])
        ).cuda()
        batch_traj_img_fts = pad_tensors(sum(batch_traj_img_fts, [])).cuda()
        batch_traj_depth_fts = pad_tensors(sum(batch_traj_depth_fts, [])).cuda()
        batch_traj_loc_fts = pad_tensors(sum(batch_traj_loc_fts, [])).cuda()

        # gmap
        batch_gmap_lens = torch.LongTensor([len(x) for x in batch_gmap_step_ids]).cuda()
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()    

        return {
            'txt_lens': batch_txt_len, 'txt_ids': batch_txt_id,
            'traj_step_lens': batch_traj_step_lens, 'traj_vp_view_lens': batch_traj_vp_view_lens, 
            'traj_view_img_fts': batch_traj_img_fts, 'traj_view_dep_fts': batch_traj_depth_fts,
            'traj_loc_fts': batch_traj_loc_fts, 'gmap_lens': batch_gmap_lens,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts
        }
    

    def _nav_gmap_distance_variable(self, txt_id, cur_vp, cur_pos, cur_ori):
        batch_txt_id = []
        batch_gmap_step_ids, batch_gmap_lens, batch_gmap_pos_fts = [], [], []
        batch_traj_img_fts, batch_traj_depth_fts, batch_traj_loc_fts = [], [], []

        for i, gmap in enumerate(self.gmaps):
            # instruction
            txt_ids = torch.LongTensor(txt_id[i])

            # traj features
            node_vp_ids = list(gmap.node_pos.keys())
            node_vp_ids = node_vp_ids[-3:]

            traj_img_fts = [gmap.get_node_rgb_embeds(vp) for vp in node_vp_ids]
            traj_depth_fts = [gmap.get_node_depth_embeds(vp) for vp in node_vp_ids]
            traj_loc_fts = [torch.from_numpy(gmap.node_loc[vp]) for vp in node_vp_ids]

            # gmap features
            gmap_step_ids = [id for id in range(1, len(node_vp_ids)+1)]
            gmap_pos_fts = torch.from_numpy(
                gmap.get_pos_fts(cur_vp[i], cur_pos[i], cur_ori[i], node_vp_ids)
            )

            batch_txt_id.append(txt_ids)
            batch_traj_img_fts.append(traj_img_fts)
            batch_traj_depth_fts.append(traj_depth_fts)
            batch_traj_loc_fts.append(traj_loc_fts)

            batch_gmap_pos_fts.append(gmap_pos_fts)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
     
        # collate
        # instruction
        batch_txt_len = torch.LongTensor([len(x) for x in batch_txt_id]).cuda()
        batch_txt_id = pad_sequence(batch_txt_id, batch_first=True, padding_value=0).cuda()
 
        # trajectory
        batch_traj_step_lens = [len(x) for x in batch_traj_img_fts]
        batch_traj_vp_view_lens = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch_traj_img_fts], [])
        ).cuda()
        batch_traj_img_fts = pad_tensors(sum(batch_traj_img_fts, [])).cuda()
        batch_traj_depth_fts = pad_tensors(sum(batch_traj_depth_fts, [])).cuda()
        batch_traj_loc_fts = pad_tensors(sum(batch_traj_loc_fts, [])).cuda()

        # gmap
        batch_gmap_lens = torch.LongTensor([len(x) for x in batch_gmap_step_ids]).cuda()
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        
        return {
            'txt_lens': batch_txt_len, 'txt_ids': batch_txt_id,
            'traj_step_lens': batch_traj_step_lens, 'traj_vp_view_lens': batch_traj_vp_view_lens, 
            'traj_view_img_fts': batch_traj_img_fts, 'traj_view_dep_fts': batch_traj_depth_fts,
            'traj_loc_fts': batch_traj_loc_fts, 'gmap_lens': batch_gmap_lens,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts
        }
   
    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap, no_vp_left) in enumerate(zip(self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(pos, real_pos) for vp, [pos, real_pos] in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp_pos = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(target_ghost_vp_pos)
                elif self.config.IL.expert_policy == 'ndtw':
                    target_ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(target_ghost_vp_pos)
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()


    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward() # self.loss.backward()
            self.scaler.step(self.optimizer)        # self.optimizer.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        print(f"waypoint_spacing:{self.policy.net.waypoint_spacing}")
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            episode_info = self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        # print(f'self.world_size:{self.world_size}')
        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            print('start evaluation!')
            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def get_ori(self):
        pos_ori = self.envs.call(['get_ori']*self.envs.num_envs)
        ori = [x for x in pos_ori]
        return ori
    
    def cal_distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)

        return xz_dist

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            curr_eps = self.envs.current_episodes()
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        unpad_txt_ids = []
        for i in range(len(observations)):
            txt_ids = [id for id, pad in zip(all_txt_ids[i],all_txt_masks[i]) if pad]
            unpad_txt_ids.append(txt_ids)
        

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        self.gmaps = [GraphMap()]
        prev_vp = [None] * self.envs.num_envs


        self.action_execution_horizon = self.policy.net.action_execution_horizon
        self.waypoint_spacing = self.policy.net.waypoint_spacing

        episode_info=[]

        for stepk in range(0, self.max_len, self.action_execution_horizon): 
            total_actions += self.envs.num_envs
            dp_txt_ids = unpad_txt_ids[:self.envs.num_envs]

            # cand waypoint prediction
            obs_outputs = self.policy.net(
                mode = "observation",
                observations = batch,
            )

            # get vp_id, vp_pos
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp = []
            for i in range(self.envs.num_envs):
                cur_vp_i = self.gmaps[i].identify_node()
                cur_vp.append(cur_vp_i)

            # save obs to gmap
            for i in range(self.envs.num_envs):
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], obs_outputs['pano_rgb'][i], obs_outputs['pano_depth'][i].squeeze(-1))

            # gen observation conditon for diffusion policy
            obs_batch = self._nav_gmap_variable(dp_txt_ids, cur_vp, cur_pos, cur_ori)
            obs_inputs = {
                'mode': 'observation_condition',
                'batch': obs_batch,
            }
            observation_condition = self.policy.net(**obs_inputs)

            # diffusion policy to navigation
            nav_inputs = {
                'mode': 'navigation',
                'observation_conditions': observation_condition,
                'observations': cur_vp,
            }

            # un-nomalize
            gc_actions = self.policy.net(**nav_inputs)
            gc_actions = gc_actions * self.waypoint_spacing/(4.0)
            vln_actions = torch.zeros(*gc_actions.shape[:2], 3).to(self.device)
            for i in range(self.envs.num_envs):
                vln_actions[i] = calculate_world_coordinates(torch.from_numpy(cur_pos[i]).to(self.device),
                                                             cur_ori[i], gc_actions[i])
            vln_actions = vln_actions.cpu().numpy()

            # stop predict
            stop_cond = {
                'mode': 'distance',
                'observation_conditions': observation_condition,
            }
            stop_logits = self.policy.net(**stop_cond)
            stop_prob = torch.sigmoid(stop_logits)
            stop_threshold = 0.10
            stop = (stop_prob < stop_threshold).cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            update_gmap = []
            for i, gmap in enumerate(self.gmaps):
                if stepk == self.max_len - 1 or stop:
                    exec_ations = vln_actions[i][:self.action_execution_horizon]
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            # 'ghosts': ghost_nodes[i],
                            'predict_ghost': list(exec_ations),
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'exec_ations': exec_ations,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                            'waypoint_spacing': self.waypoint_spacing,
                        }
                    )
                else:
                    exec_ations = vln_actions[i][:self.action_execution_horizon]
                    if self.config.VIDEO_OPTION:
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            # 'ghosts': ghost_nodes[i],
                            'predict_ghost': list(exec_ations),
                        }
                    else:
                        vis_info = None

                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'exec_ations': exec_ations,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                            'waypoint_spacing': self.waypoint_spacing,
                        }
                    )
                    update_gmap.append(i)

                prev_vp[i] = str(len(gmap.node_pos) - 1)

            outputs = self.envs.step(env_actions)
            observations, agent_states, dones, infos = [list(x) for x in zip(*outputs)]

            # update gmap
            for i in update_gmap:
                gmap_observations = observations[i]
                gmap_agent_states = agent_states[i]
                for j in range(len(gmap_observations)):
                    if j == len(gmap_observations) - 1:
                        observations[i] = gmap_observations[j]
                        break

                    gmap_observation = extract_instruction_tokens([gmap_observations[j]],self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
                    batch = batch_obs(gmap_observation, self.device)
                    batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                    # cand waypoint prediction
                    obs_outputs = self.policy.net(
                        mode = "observation",
                        observations = batch,
                    )

                    # get vp_id, vp_pos
                    cur_pos, cur_ori = gmap_agent_states[j].position, gmap_agent_states[j].rotation
                    cur_vp_i = self.gmaps[i].identify_node()

                    # save obs to gmap
                    self.gmaps[i].update_graph(prev_vp[i], stepk+1+j,
                                            cur_vp_i, cur_pos, obs_outputs['pano_rgb'][0], obs_outputs['pano_depth'][0])
                    prev_vp[i] = str(len(self.gmaps[i].node_pos) - 1)
            

            for i in range(self.envs.num_envs):
                episode_info.append([curr_eps[i].episode_id, curr_eps[i].goals])
                    

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']

                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
        
        return episode_info