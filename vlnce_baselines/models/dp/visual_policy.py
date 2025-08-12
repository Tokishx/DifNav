import torch
import torch.nn as nn
from vlnce_baselines.models.dp.model.vilmodel import CrossAttentionQFormer
from vlnce_baselines.models.dp.model.pretrain_cmt import GlocalTextPathCMTPreTraining

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

class Visual_policy(nn.Module):
    def __init__(
            self, config, xlbert: GlocalTextPathCMTPreTraining,
    ) -> None:
        super().__init__()
        self.bert = xlbert.bert

        # Q-Former
        self.do_compression = True
        self.qFormer = CrossAttentionQFormer(config)

        # hidden_dim to encoding_size
        self.denseLayer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, batch):

        txt_ids = batch['txt_ids']
        txt_lens = batch['txt_lens']
        traj_view_img_fts = batch['traj_view_img_fts']
        traj_view_dep_fts = batch['traj_view_dep_fts']
        traj_loc_fts = batch['traj_loc_fts']
        traj_step_lens = batch['traj_step_lens']
        traj_vp_view_lens = batch['traj_vp_view_lens']
        nav_step_ids = batch['gmap_step_ids']
        nav_pos_fts = batch['gmap_pos_fts']
        gmap_lens = batch['gmap_lens']

        nav_conditional_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_view_dep_fts, traj_loc_fts, 
            traj_step_lens, traj_vp_view_lens, nav_step_ids, nav_pos_fts, gmap_lens
        )  

        # for diffusion_transformer # should be marked
        nav_conditional_embeds = nav_conditional_embeds.squeeze(dim=1)

        # # Q-former
        # if self.do_compression:
        #     # [bs, query_nums, hidden_size]
        #     nav_conditional_embeds = self.qFormer(nav_conditional_embeds)
        return nav_conditional_embeds


