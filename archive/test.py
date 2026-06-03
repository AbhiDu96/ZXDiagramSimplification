from zx_env import zx_env
from ppo import make_env
import torch
import numpy as np
import hydra
from MCTS import MCTS_like, PPO_update, process_MCTS
from model import MCTS_like_model
from omegaconf import DictConfig, OmegaConf, read_write
import ray
import os
from torch.utils.tensorboard import SummaryWriter
from time import time

torch.manual_seed(1)
np.random.seed(1)

device = "cpu"

def get_next(remaining_refs):
    while len(remaining_refs)> 0:
        ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1, timeout=None)
        yield ray.get(ready_refs[0])

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg):
    with read_write(cfg):
        cfg.batch_size = int(cfg.env.num_envs * cfg.algorithm.num_steps)
        cfg.minibatch_size = int(cfg.batch_size // cfg.algorithm.num_minibatches)
        cfg.num_iterations = cfg.algorithm.total_timesteps // cfg.batch_size
    run_name = f"QACI_graph_optim__{cfg.exp_name}__{cfg.seed}__{int(time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg,resolve=True).items()])),
    )


    model = MCTS_like_model(
        cfg.model.action_dim,
        make_env(cfg,"cpu").action_space.n,  # type: ignore
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_message_passing=cfg.model.n_message_passing,
        device=device,
        model_type=cfg.model.model_type,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    remaining_steps = cfg.algorithm.total_timesteps
    global_step=0
    while remaining_steps>=0:
        global_step+=1
        print("starting mcts. Remaining steps",remaining_steps)
    
        logs = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        actss = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        indices = torch.zeros(cfg.env.num_envs, cfg.algorithm.num_steps)-5
        advs = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        rets = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        vss = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        rewards = torch.empty(cfg.env.num_envs, cfg.algorithm.num_steps)
        roots = np.empty((cfg.env.num_envs, cfg.algorithm.num_steps), dtype=object)
        ages = torch.ones((cfg.env.num_envs,cfg.algorithm.num_steps))*(-1)
        mcts = [
            process_MCTS.remote(cfg,model)
            #process_MCTS(cfg,model)
            for _ in range(cfg.env.num_envs)
        ]
        mcts = get_next(mcts)
        print("done mcts")
        with torch.no_grad():
            for i, m in enumerate(mcts):
                #out = m.rollout(cfg.algorithm.num_steps)
                #root, logp, acts, adv, ret, vs = m.extract_data(*out)
                root, logp, acts,idx, adv, ret, vs,rews = m
                remaining_steps-=len(logp)
                logs[i,:len(logp)] = torch.from_numpy(logp)
                actss[i,:len(acts)] = torch.from_numpy(acts)
                indices[i,:len(idx)]=torch.from_numpy(idx)
                advs[i,:len(adv)] = torch.from_numpy(adv)
                rets[i,:len(ret)] = torch.from_numpy(ret)
                vss[i,:len(vs)] = torch.from_numpy(vs)
                rewards[i,:len(rews)]=torch.from_numpy(rews)
                roots[i] = root
                ages[i,:len(logp)] = torch.arange(len(logp))
        model.to(device)

        v_loss, pg_loss, entropy_loss = PPO_update(
            cfg,
            model,
            optimizer,
            roots.reshape(-1).tolist(),
            logs.reshape(-1).to(device),
            actss.reshape(-1).to(device),
            indices.reshape(-1).to(device),
            advs.reshape(-1).to(device),
            rets.reshape(-1).to(device),
            vss.reshape(-1).to(device),
            ages.reshape(-1).to(device),
        )
        model.to("cpu")
        y_pred, y_true = vss.reshape(-1).cpu().numpy(), rets.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/episodic_return", rets[:,-1].mean(), global_step
        )
        writer.add_scalar("chart/total_reward",rewards.sum(1).mean(),global_step)




if __name__ == "__main__":
    ray.init(address=os.environ["ip_head"])
    print("environment",os.environ["ip_head"])
    print("Nodes in the Ray cluster:")
    print(ray.nodes())
    main()
