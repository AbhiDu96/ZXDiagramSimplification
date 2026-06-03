# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
import logging
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from models import ActionModel
from torch_geometric.data import Batch
from zx_env import zx_env
import hydra
from omegaconf import DictConfig, OmegaConf, read_write
import utils
from tqdm import trange
from multiEnv import ParallelVecEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(cfg, device):
    env = zx_env(
        mutate_probability = cfg.env.mutate_prob,
        mutation_steps = cfg.env.mutation_steps,
        reward_fn = cfg.env.reward_fn,
        n_qubits=cfg.env.n_qubits,
        depth = cfg.env.depth,
        mq_ratio=cfg.env.cnot_fraction,
        t_ratio=cfg.env.t_fraction,
        h_ratio=cfg.env.h_fraction,
        max_steps= cfg.env.max_env_steps,
        negative_reward_mean= cfg.env.extra_noise_mean,
        negative_reward_std=cfg.env.extra_noise_std,
        rules_list=cfg.env.rules_used,
        full_fuse_every_step=cfg.env.full_fuse_every_step,
        reduce_at_reset=cfg.env.reduce_at_reset
    )
    if cfg.env.sparsify_reward:
        env = utils.RewardTransform(env)
    if cfg.env.make_directed:
        env = utils.GraphMakeDirected(env)
    env = utils.GraphMaskWrapper(env,device)
    return env

def validate(cfg,global_step,writer,val_env,zx_graph, agent):
    logging.info("STARTING VALIDATION")
    next_obs,_ = val_env.reset(initial_circuit_graph=zx_graph)
    returns = torch.zeros(cfg.env.num_envs)
    for step in range(0, cfg.env.max_env_steps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            # make a temporary batch from our data:
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = val_env.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        returns = returns+torch.tensor(reward).view(-1)
        next_obs, next_done = (
            next_obs,
            torch.Tensor(next_done),
        )
    writer.add_scalar("valid/reward",returns.mean(),global_step)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def step_envs(envs: [gym.Env], actions: [(int, int)]):
    next_obs, reward, terminations, truncations, infos = (
        [],
        [],
        [],
        [],
        [],
    )

    for e, a in zip(envs, actions):
        pos,ac = a.astype(np.int32)
        logging.debug("position %s action %s", pos, ac)
        o, r, t1, t2, i = e.step(ac,position=pos)
        next_obs.append(o)
        reward.append(r)
        terminations.append(t1)
        truncations.append(t2)
        if t1 or t2:
            infos.append(i)
    return next_obs, reward, terminations, truncations, infos


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    logging.info("config: %s", OmegaConf.to_yaml(cfg))
    #torch.autograd.set_detect_anomaly(True)
    with read_write(cfg):
        cfg.batch_size = int(cfg.env.num_envs * cfg.algorithm.num_steps)
        cfg.minibatch_size = int(
            cfg.batch_size // cfg.algorithm.num_minibatches
        )
        cfg.num_iterations = cfg.algorithm.total_timesteps // cfg.batch_size
    run_name = f"QACI_graph_optim__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg,resolve=True).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )

    # env setup
    envs = utils.SingleVecEnv([make_env(cfg,device) for i in range(cfg.env.num_envs)],cfg)
    validation_envs = utils.SingleVecEnv([make_env(cfg,device) for i in range(cfg.env.num_envs)],cfg)
    _ = validation_envs.reset()
    validation_circuits = [e.state_zx_graph for e in validation_envs]


    agent = torch.compile(ActionModel(
        cfg.model.action_dim,
        envs[0].action_space.n,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_message_passing=cfg.model.n_message_passing,
        device=device,
        model_type = cfg.model.model_type,
    ).to(device),mode="max-autotune",dynamic=True)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.algorithm.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = np.empty((cfg.algorithm.num_steps, cfg.env.num_envs), dtype=object)
    # this stores both the action and position on the augmented graph
    actions = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs, 2),device=device)
    logprobs = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs),device=device)
    rewards = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs),device=device)
    dones = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs),device=device)
    values = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs),device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(cfg.env.num_envs)

    for iteration in range(1, cfg.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if cfg.algorithm.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.algorithm.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.algorithm.num_steps):
            global_step += cfg.env.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # make a temporary batch from our data:
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob.detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).view(-1)
            next_obs, next_done = (
                next_obs,
                torch.Tensor(next_done),
            )
            logging.debug("reward: %.4f", torch.tensor(reward).float().mean())

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logging.info("global_step=%d, episodic_return=%.4f", global_step, info["episode"]["r"])
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.algorithm.num_steps)):
                if t == cfg.algorithm.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + cfg.algorithm.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + cfg.algorithm.gamma
                    * cfg.algorithm.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        approx_kl=0
        old_approx_kl=0

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.algorithm.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                t0 = time.time()
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.algorithm.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.algorithm.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.algorithm.clip_coef, 1 + cfg.algorithm.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.algorithm.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.algorithm.clip_coef,
                        cfg.algorithm.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - cfg.algorithm.ent_coef * entropy_loss
                    + v_loss * cfg.algorithm.vf_coef
                )
                t1 = time.time()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), cfg.algorithm.max_grad_norm
                )
                optimizer.step()
                logging.debug("forward=%.3fs backward+update=%.3fs", t1 - t0, time.time() - t1)

            if (
                cfg.algorithm.target_kl is not None
                and approx_kl > cfg.algorithm.target_kl
            ):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        logging.info("SPS: %d", sps)
        writer.add_scalar("charts/SPS", sps, global_step)
        if iteration % 10 == 1:
            validate(cfg,global_step,writer,validation_envs,validation_circuits,agent)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
    pass
