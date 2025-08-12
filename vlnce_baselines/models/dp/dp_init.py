import torch
import yaml
import os
from vlnce_baselines.models.dp.nomad import NoMaD, DenseNetwork
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from vlnce_baselines.models.dp.model.pretrain_cmt import GlocalTextPathCMTPreTraining
from vlnce_baselines.models.dp.model.vilmodel import StopPrediction
from vlnce_baselines.models.dp.visual_policy import Visual_policy

from transformers import PretrainedConfig

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import numpy as np
import torch.backends.cudnn as cudnn



def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vln_nomad_models(config=None):
    model_config = config.DP.model_config
    pretrained_path_ol = config.pretrained_path_ol
    load_from_iters = False

    default_config = config.DP.default_config
    nomad_config = config.DP.nomad_config

    with open(default_config, "r") as f:
        default_nomad_config = yaml.safe_load(f)

    config = default_nomad_config


    with open(nomad_config, "r") as f:
        user_config = yaml.safe_load(f)
    
    config.update(user_config)

    if os.path.exists(pretrained_path_ol):
        print(f"load from iters: {pretrained_path_ol}")
        load_from_iters = True


    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    model_config = PretrainedConfig.from_json_file(model_config)
    vision_model = GlocalTextPathCMTPreTraining(model_config)

    # nomad model load
    vision_encoder = Visual_policy(
                model_config, vision_model
            )

    noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
    
    # # diffusion_transformer
    # noise_pred_net = TransformerForDiffusion(
    #         input_dim=config['input_dim'],
    #         output_dim=config['output_dim'],
    #         horizon=config['horizon'],
    #         n_obs_steps=config['n_obs_steps'],
    #         cond_dim=config['cond_dim'],
    #         causal_attn=config['causal_attn'],
    #         n_layer=config['n_layer'],
    #         n_head=config['n_head'],
    #         n_emb=config['n_emb'],
    #         p_drop_emb=config['p_drop_emb'],
    #         p_drop_attn=config['p_drop_attn'],
    #     )        

    dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(config["pretrained_path"], map_location=device)
    print(f"load model from: {config['pretrained_path']}")

    model.load_state_dict(state_dict)

    # # 检查model2是否受到影响
    # for param1, param2 in zip(model.parameters(), distance_model.parameters()):
    #     assert not torch.equal(param1, param2), "Model2's parameters were affected by Model1!"
    # # 训练时不适用ema_model，只有在eval时使用
    # ema_model = EMAModel(model=model,power=0.75)
    # ema_model = ema_model.averaged_model
    # ema_model.train()
    # ema_model.eval()

    

    # 冻结所有参数，不可训练
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    len_traj_pred = config["len_traj_pred"]
    action_dim = config["action_dim"]
    action_execution_horizon = config["action_execution_horizon"]

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "waypoint_spacing" not in data_config:
            raise ValueError
        elif "metric_waypoint_spacing" not in data_config:
            raise ValueError
            # data_config["waypoint_spacing"] = 1
    waypoint_spacing = data_config["waypoint_spacing"]
    metric_waypoint_spacing = data_config["metric_waypoint_spacing"]

    params = {
        'len_traj_pred': len_traj_pred,
        'action_dim': action_dim,
        'action_execution_horizon': action_execution_horizon,
        'waypoint_spacing': waypoint_spacing,
        'metric_waypoint_spacing': metric_waypoint_spacing,
    }

        
    return model, None, None, noise_scheduler, params


if __name__ == "__main__":

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    nomad_config = 'config/nomad.yaml'

    with open(nomad_config, "r") as f:
        user_config = yaml.safe_load(f)
    
    config.update(user_config)

    model_config = 'run_pt/r2r_model_config_dep.json'

    model = get_vlnbert_models(config=config, model_config=model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

