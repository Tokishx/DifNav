from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.dp.dp_init import get_vln_nomad_models
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
from vlnce_baselines.models.dp.utils.visualize_utils import from_numpy, to_numpy
import math
from gym.spaces import Box, Dict
import cv2

with open(os.path.join(os.path.dirname(__file__), "dp/config/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
ACTION_STATS = {}
for key in data_config['action_stats']:
    ACTION_STATS[key] = np.array(data_config['action_stats'][key])

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def build_feature_extractor(model_name, checkpoint_file, device):
    # model, preprocess = clip.load(model_name, device='cuda')
    box_space = Box(low=0.0, high=1.0, shape=(256, 256, 1), dtype='float32')

    obs_space = Dict({"depth": box_space})
    model = VlnResnetDepthEncoder(
        obs_space,
        output_size=256,
        checkpoint=checkpoint_file,
        backbone=model_name,
        spatial_output=False,
    ).to(device)
    model.eval()
    return model

@baseline_registry.register_policy
class PolicyViewSelectionETPDP(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            ETP(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class ETP(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print('\nInitalizing the VLN_NOMAD model ...')
        # self.vln_bert = get_vlnbert_models(config=model_config)
        self.nomad, _, _, self.noise_scheduler, dp_params = get_vln_nomad_models(config=model_config)
        self.nomad_global = None
        self.action_dim = dp_params["action_dim"]
        self.pred_horizon = dp_params["len_traj_pred"]
        self.action_execution_horizon = dp_params["action_execution_horizon"]
        self.waypoint_spacing = dp_params["waypoint_spacing"]
        
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(2048, 768)
        # elif model_config.task_type == 'rxr':
        #     self.rgb_projection = nn.Linear(2048, 512)
        # self.rgb_projection = nn.Linear(2048, 768) # for vit 768 compability
        # if model_config.task_type == 'r2r':
        #     self.rgb_projection = nn.Linear(512, 768)
        # else:
        #     self.rgb_projection = None
        self.drop_env = nn.Dropout(p=0.4)

        # self.pos_encoder = nn.Sequential(
        #     nn.Linear(6, 768),
        #     nn.LayerNorm(768, eps=1e-12)
        # )
        # self.hist_mlp = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        # )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        ).to(self.device)
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))

        # Init the RGB encoder
        # assert model_config.RGB_ENCODER.cnn_type in [
        #     "TorchVisionResNet152", "TorchVisionResNet50"
        # ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        # if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
        #     self.rgb_encoder = TorchVisionResNet50(
        #         observation_space,
        #         model_config.RGB_ENCODER.output_size,
        #         device,
        #         spatial_output=model_config.spatial_output,
        #     )
        self.rgb_encoder = CLIPEncoder(self.device)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, mode=None, observations=None,
                batch=None, observation_conditions=None, global_observation_conditions=None):

        if mode == 'observation':

            batch_size = observations['rgb'].shape[0]
            ''' encoding rgb/depth at all directions ----------------------------- '''
            NUM_IMGS = 12
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if 'depth' in k:  # You might need to double check the keys order
                    for bi in range(v.size(0)):
                        # key order for vln_nomad
                        ra_count = a_count % NUM_IMGS
                        depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12['depth'] = depth_batch
            obs_view12['rgb'] = rgb_batch
            depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])
            rgb_embedding = self.rgb_encoder(obs_view12)      # torch.Size([bs, 2048, 7, 7])

            rgb_embed_reshape = rgb_embedding.reshape(
                batch_size, NUM_IMGS, 512, 1, 1)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4)
            
            rgb_feats = self.space_pool_rgb(rgb_embed_reshape)    # torch.Size([bs, 12， 512])
            depth_feats = self.space_pool_depth(depth_embed_reshape) # torch.Size([bs, 12， 128])

            outputs = {
                'pano_rgb': rgb_feats,
                'pano_depth': depth_feats,
            }
            
            return outputs

        elif mode == 'observation_condition':

            outs = self.nomad(
                "vision_encoder", batch=batch
            )
            return outs

        elif mode == 'global_observation_condition':

            outs = self.nomad_global(
                "vision_encoder", batch=batch
            )
            return outs


        elif mode == 'distance_observation_condition':
            outs = self.nomad(
                "vision_encoder", batch=batch
            )
            return outs

        elif mode == 'navigation':

            noisy_diffusion_output = torch.randn(
                (len(observations), self.pred_horizon, self.action_dim), device=self.device)

            diffusion_output = noisy_diffusion_output

            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                noise_pred = self.nomad(
                    "noise_pred_net",
                    sample=diffusion_output,
                    timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device),
                    global_cond=observation_conditions
                )

                # inverse diffusion step (remove noise)
                diffusion_output = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=diffusion_output
                ).prev_sample
            # gc_actions
            outs = get_action(diffusion_output, ACTION_STATS)

            return outs

        elif mode == 'distance':
            outs = self.nomad(
                'dist_pred_net', obsgoal_cond=observation_conditions
            )

            return outs

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
