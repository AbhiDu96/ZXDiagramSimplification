import hydra
from omegaconf import DictConfig, OmegaConf, read_write
import utils
import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from TreePolicy import Tree, start_tree, extract_optimal_path, synchronous_forward
import random
from torch.utils.tensorboard import SummaryWriter
from ppo import make_env
from torch import optim
import time
from tqdm import trange
from model import BundleNet
import pickle
import os
import ray
from zx_env import extract_circuit


def restart_tree(cfg, tree : Tree):
    idx = np.argmax(tree.rewards)
    best_node = tree.nodes[idx]
    best_zx = tree.zx_states[idx]
    best_info = tree.infos[idx]
    return start_tree(cfg, best_node,best_zx,best_info)

@ray.remote
def deploy_train(cfg, agent):
    agent.eval()
    env = make_env(cfg, "cpu")
    # ALGO Logic: Storage setup
    obs = np.empty((cfg.algorithm.num_steps,), dtype=object)
    # this stores both the action and position on the augmented graph
    logprobs = torch.zeros((cfg.algorithm.num_steps,))
    rewards = torch.zeros((cfg.algorithm.num_steps,))
    dones = torch.zeros((cfg.algorithm.num_steps,))
    values = torch.zeros((cfg.algorithm.num_steps,))
    next_obs, infos = env.reset()
    next_obs = start_tree(cfg,next_obs.Graph, next_obs.state_zx_graph, info=infos)
    actions = torch.zeros((cfg.algorithm.num_steps, 1+cfg.multi_range))
    next_done = 0
    for step in range(0, cfg.algorithm.num_steps):
        print(next_done)
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            # make a temporary batch from our data:
            effect, total_log, _, value = next_obs.select(agent)
            values[step] = torch.tensor(value).flatten()
        #print("Efffect output",effect,actions.shape,next_obs.multi_range)
        actions[step] = torch.tensor(effect)
        logprobs[step] = torch.tensor(total_log).detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
        if step % cfg.max_tree_size == cfg.max_tree_size-1:
            new_tree = restart_tree(cfg,new_tree)
        next_done = np.logical_or(term, trunc)
        rewards[step] = torch.tensor(rew).view(-1)
        next_obs, next_done = (
            new_tree,
            torch.Tensor([next_done]).squeeze(),
        )
    agent.train()
    return obs, actions, logprobs, rewards, dones, values, next_obs, next_done


@ray.remote
def deploy_agents(cfg, state, agent):
    env = make_env(cfg, "cpu")
    next_obs, info = env.reset(initital_circuit_graph=state)
    next_obs = start_tree(cfg,next_obs.Graph, next_obs.state_zx_graph, info=info)
    rewards = 0
    for step in range(0, cfg.algorithm.num_steps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            # make a temporary batch from our data:
            effect, total_log, _, value = next_obs.select(agent)

        # TRY NOT TO MODIFY: execute the game and log data.
        new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
        if step % cfg.max_tree_size == cfg.max_tree_size-1:
            new_tree = restart_tree(cfg,new_tree)
        next_done = np.logical_or(term, trunc)
        next_obs, next_done = (
            new_tree,
            torch.Tensor(next_done),
        )
        rewards += rew
    return rewards, next_obs


def validation(run_name, global_step, cfg, writer, validations, agent):
    print("STARTING VALIDATION")
    # next_obs, _ = zip(*[e.reset(initital_circuit_graph=v) for e,v in zip(envs,validations)])
    res = []
    for v in validations:
        res.append(deploy_agents.remote(cfg, v, agent))
    rewards, next_obs = zip(*ray.get(res))
    tq = [
        extract_circuit(circuit.get_best_node())[0].stats_dict()["twoqubit"]
        for circuit in next_obs
    ]
    writer.add_scalar(
        "charts/validation_return", torch.tensor(rewards).mean().item(), global_step
    )
    writer.add_scalar("charts/validation_mean_CNOT", np.mean(tq), global_step)
    writer.add_scalar("charts/validation_max_CNOT", np.max(tq), global_step)
    writer.add_scalar("charts/validation_min_CNOT", np.min(tq), global_step)
    print([circuit.get_best_info() for circuit in next_obs])
    level = [circuit.get_best_info()["level"] for circuit in next_obs]
    writer.add_scalar("charts/expected_level", np.mean(level), global_step)
    optimal_paths = [extract_optimal_path(t) for t in next_obs]

    if not os.path.exists(f"runs/{run_name}/saves"):
        os.mkdir(f"runs/{run_name}/saves")
    # print("lsdir",os.listdir(f"runs/{run_name}/saves"))
    with open(f"runs/{run_name}/saves/data-{global_step}.pkl", "wb+") as writer:
        pickle.dump(optimal_paths, writer)
    torch.save(agent.state_dict(), f"runs/{run_name}/saves/model-{global_step}.pth")
    #prune_old_models(run_name, 3)
    print(
        "VALIDATION DONE",
    )
    # return best_node


def prune_old_models(run_name, keep):
    path = f"runs/{run_name}/saves/"
    folder = os.listdir(path)
    folder = sorted(folder, key=lambda x: int(x.split(".")[0].split("-")[-1]))[-keep:]
    for r in folder:
        os.remove(path + r)


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    print(cfg)
    # torch.autograd.set_detect_anomaly(True)
    with read_write(cfg):
        cfg.batch_size = int(cfg.env.num_envs * cfg.algorithm.num_steps)
        cfg.minibatch_size = int(cfg.batch_size // cfg.algorithm.num_minibatches)
        cfg.num_iterations = cfg.algorithm.total_timesteps // cfg.batch_size
    run_name = f"QACI_graph_optim__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{value}|"
                    for key, value in OmegaConf.to_container(cfg, resolve=True).items()
                ]
            )
        ),
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
    validation_circuits = []
    envs = [make_env(cfg, device) for i in range(cfg.env.num_envs)]
    # validation_envs = [make_env(cfg,device) for i in range(cfg.env.num_envs)]
    # _ = zip(*[v.reset()  for v in validation_envs])
    for _ in range(cfg.n_validation):
        e = make_env(cfg, device)
        e.reset()
        validation_circuits.append(e.state_zx_graph_initital)
    # validation_circuits = [e.state_zx_graph for e in validation_envs]

    agent = BundleNet(
        cfg.model.action_dim,
        envs[0].action_space.n,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_message_passing=cfg.model.n_message_passing,
        device="cpu",
        model_type=cfg.model.model_type,
    )  # ,mode="reduce-overhead",dynamic=True)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.algorithm.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = np.empty((cfg.algorithm.num_steps, cfg.env.num_envs), dtype=object)
    # this stores both the action and position on the augmented graph
    actions = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs, 1+cfg.multi_range))
    logprobs = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs))
    rewards = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs))
    dones = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs))
    values = torch.zeros((cfg.algorithm.num_steps, cfg.env.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_done = torch.zeros(cfg.env.num_envs)
    start_time = time.time()

    for iteration in range(1, cfg.num_iterations + 1):
        # running validation:
        if iteration % 50 == 1:
            validation(run_name, global_step, cfg, writer, validation_circuits, agent)
        agent = agent.cpu()
        # Annealing the rate if instructed to do so.
        if cfg.algorithm.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.algorithm.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        """next_obs, infos = zip(*[e.reset() for e in envs])
        next_obs = [start_tree(n.Graph,n.state_zx_graph,info=info) for n,info in zip(next_obs,infos)]
        for step in range(0, cfg.algorithm.num_steps):
            global_step += cfg.env.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # make a temporary batch from our data:
                effect, total_log, _, value = zip(*[n.select(agent) for n in next_obs])
                values[step] = torch.tensor(value).flatten()
            actions[step] = torch.tensor(effect)
            logprobs[step] = torch.tensor(total_log).detach()

            # TRY NOT TO MODIFY: execute the game and log data.
            new_tree, rew, term, trunc, info = zip(*[n.expand(a,e) for e,a,n in zip(envs,effect,next_obs)])
            next_done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(rew).view(-1)
            next_obs, next_done = (
                new_tree,
                torch.Tensor(next_done),
            )
            print("reward",torch.tensor(rewards[step]).float().mean())"""
        global_step += cfg.env.num_envs * cfg.algorithm.num_steps
        next_obs, next_done = (
            np.empty(cfg.env.num_envs, dtype=object),
            torch.empty(cfg.env.num_envs),
        )
        rollouts = []
        for i in range(cfg.env.num_envs):
            rollouts.append(deploy_train.remote(cfg, agent))
        for i, rollout in enumerate(rollouts):
            (
                obs[:, i],
                actions[:, i],
                logprobs[:, i],
                rewards[:, i],
                dones[:, i],
                values[:, i],
                next_obs[i],
                next_done[i],
            ) = ray.get(rollout)
        writer.add_scalar(
            "charts/episodic_return", rewards.sum(0).mean().item(), global_step
        )
        # bootstrap value if not done
        with torch.no_grad():
            next_value = torch.tensor(
                list(zip(*[n.select(agent) for n in next_obs]))[-1]
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
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
        b_logprobs = logprobs.reshape(-1).to(device)
        b_actions = actions.reshape((-1, 1+cfg.multi_range)).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)
        b_values = values.reshape(-1).to(device)
        agent = agent.to(device)
        approx_kl = 0
        old_approx_kl = 0

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        print("Running iteration", iteration)
        for epoch in trange(cfg.algorithm.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                # t0 = time.time()
                # print("Start")
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]
                t0 = time.time()
                cache, newlogprob, entropy,newvalue = synchronous_forward(
                    cfg, b_obs[mb_inds], agent, b_actions.long()[mb_inds]
                )
                t1 = time.time()
                # _, newlogprob, entropy, newvalue = zip(*[b.select(agent,a) for b,a in zip(b_obs[mb_inds],b_actions.long()[mb_inds])])
                print("length", len(cache))
                """_, newlogprob, entropy, newvalue = zip(
                    *[
                        b.select(agent, a, device=device, cache=c)
                        for c, b, a in zip(
                            cache, b_obs[mb_inds], b_actions.long()[mb_inds]
                        )
                    ]
                newlogprob = torch.stack(newlogprob)
                entropy = torch.stack(entropy)
                newvalue = torch.stack(newvalue)
                )"""
                t3 = time.time()
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
                print("losses", pg_loss, entropy_loss, v_loss, "ratio", ratio)
                loss = (
                    pg_loss
                    - cfg.algorithm.ent_coef * entropy_loss
                    + v_loss * cfg.algorithm.vf_coef
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), cfg.algorithm.max_grad_norm
                )
                optimizer.step()
                t4 = time.time()
                print(
                    "running forward",
                    t1 - t0,
                    "selection",
                    t3 - t1,
                    "optimizing",
                    t4 - t3,
                )

            if (
                cfg.algorithm.target_kl is not None
                and approx_kl > cfg.algorithm.target_kl
            ):
                break
        agent.cpu()
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
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    [e.close() for e in envs]
    writer.close()


if __name__ == "__main__":
    main()
