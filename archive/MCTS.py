import numpy as np
import torch
from typing import Any, List, Tuple
from dataclasses import dataclass, field
from utils import GraphMask
from ppo import make_env
from torch import nn
from tqdm import trange
import ray
import time

@ray.remote
def process_MCTS(cfg, model):
    print("start processing")
    with torch.no_grad():
        env = make_env(cfg, "cpu")
        mcts = MCTS_like(env, model, gamma=cfg.algorithm.gamma, lmbda=cfg.algorithm.gae_lambda)
        print("made env")
        out = mcts.rollout(cfg.algorithm.num_steps)
        print("rollout done")
        root, logp, acts,indices, adv, ret, vs,rewards = mcts.extract_data(*out)
    print("done processing")
    return root, logp.numpy(), acts.numpy(),indices.numpy(), adv.numpy(), ret.numpy(), vs.numpy(),rewards.numpy()


@dataclass
class Node:
    data: GraphMask
    W: torch.Tensor | None
    Value: torch.Tensor | None
    Reward: float
    done: bool
    children: List[Any]
    parent: Any
    Age: int
    total_W: torch.Tensor | None = None
    depth: int = 0
    ac_pos: np.ndarray = field(default_factory=lambda: np.empty(1))

    def get_choosables(self, max_age=1_000_000):
        if self.done:
            return []
        ls = []
        for a, c in enumerate(self.children):
            if c is None or c.Age > max_age:
                ls.append((self, a, self.total_W[a], self.Value[a]))  # type: ignore
                continue
            ls.extend(c.get_choosables(max_age))
        return ls

    def expand(
        self,
        obs,
        ws: torch.Tensor,
        vs: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        age: int,
    ):
        if self.children[action] is not None:
            raise Exception("ERROR, THIS NODE ALREADY EXISTS")
        n_actions = obs.action_mask.sum()
        total_W = ws + self.total_W[action]  # type: ignore
        # action-position indices
        ac_p = np.stack(np.where(obs.action_mask == 1), -1)
        assert self.Age < age, (self.Age, age)
        #print("adding new node with age",age,"and children",n_actions, "to parent",self.Age)
        self.children[action] = Node(
            data=obs,
            W=ws,
            Value=vs,
            Reward=reward,
            done=done,
            children=[None for _ in range(n_actions)],
            parent=self,
            Age=age,
            total_W=total_W,
            depth=self.depth + 1,
            ac_pos=ac_p,
        )

    def strip(
        self,
    ):
        self.W = None
        self.Value = None
        self.total_W = torch.zeros_like(self.total_W)
        for c in self.children:
            if c is not None:
                c.strip()

    def flatten_tree(self, depth_limit=1_000_000):
        ls = [self.data]
        for c in self.children:
            if c is not None and c.depth < depth_limit:
                ls.extend(c.flatten_tree(depth_limit))
        return ls

    def unflatten(self, ws, vs, i, depth_limit=1_000_000):
        assert len(ws) == len(vs)
        self.W = extract_masked(ws, i)
        self.Value = extract_masked(vs, i)
        if self.parent is not None:
            self.total_W = (
                self.W + self.parent.total_W[self.parent.children.index(self)]
            )
        for c in self.children:
            if c is not None and c.depth < depth_limit:
                i = c.unflatten(ws, vs, i + 1, depth_limit)
        return i

    def best_reward(self) -> float:
        return max(
            self.Reward,
            -100000,
            *[c.best_reward() for c in self.children if c is not None],
        )

def get_distributional_data(roots,ages,actions,i):
    # print(i,len(roots),ages)
    nodes, _, total_w, vs = zip(*roots[i].get_choosables(ages[i]))
    #print("got choosables for",i,time.time()-t0)
    total_w = torch.stack(total_w)/ (torch.tensor([n.depth for n in nodes]) + 1)
    dist = torch.distributions.Categorical(
        logits=torch.log_softmax(total_w, -1)
    )
    entropy = dist.entropy()
    newvalue = torch.max(torch.stack(vs))
    logit=torch.log_softmax(total_w,-1)[actions[i].long()]
    print(total_w)
    newlogprob = dist.log_prob(actions[i].long())
    return entropy,newvalue,newlogprob
    

@torch.no_grad()
def bootstrap_reward(rewards, values, dones, gamma, lmbda):
    rewards = rewards.to(dtype=torch.float16)
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards, dtype=torch.float16)
    lastgaelam = 0
    dones = dones.long()
    for t in reversed(range(advantages.shape[-1])):
        nextnonterminal = 1.0 - dones[t + 1]
        nextvalues = values[t + 1]
        # td-error
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        returns = rewards[t] + gamma*returns
        advantages[t] = lastgaelam = (
            delta + gamma * lmbda * nextnonterminal * lastgaelam
        )
    #returns = advantages + values[:-1]
    print(
        f"returns {returns.mean()}±{returns.std()}, advantages {advantages.mean()}±{advantages.std()}, rewards {rewards.mean()}±{rewards.std()}",
    )
    return returns, advantages

def big_step(model,roots):
    [root.strip() for root in roots]
    states = [root.flatten_tree() for root in roots]
    sizes = np.cumsum([0]+[len(state) for state in states])
    flat = [x for state in states for x in state]
    t0 = time.time()
    ws,vs= model(flat)
    t1=time.time()
    #print("model runtime",t1-t0)
    for i, root in enumerate(roots):
        root.unflatten(ws,vs,sizes[i])
    #print("model unflatten",time.time()-t1)
def PPO_update(
    cfg,
    model,
    optimizer,
    roots: List[Node],
    logprob,
    actions,
    indices,
    adv,
    returns,
    values,
    ages,
):
    b_inds = np.arange(len(returns))
    # filter out only available indices
    b_inds = b_inds[ages>=0]
    #print(b_inds)
    for epoch in trange(cfg.algorithm.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, cfg.batch_size, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            # run the tree
            mb_inds = b_inds[start:end]
            seen = list()
            unique_roots = [x for x in roots if x not in seen and not seen.append(x)]
            # print("n_roots",len(unique_roots))
            big_step(model,unique_roots)
            # now all the data is there, get the distributions
            newlogprob = torch.zeros(len(mb_inds))
            entropy = torch.zeros(len(mb_inds))
            newvalue = torch.zeros(len(mb_inds))
            for j,i in enumerate(mb_inds):
                #print("i",i,ages[i],actions[i],len(roots[i].get_choosables(ages[i])))
                newlogprob[j],entropy[j],newvalue[j]=get_distributional_data(roots,ages,indices,i)
            #for idx, i in enumerate(mb_inds):
            #print("computing values")
            print("newlogprob",newlogprob,"oldlogprob",logprob)
            logratio = newlogprob - logprob[mb_inds]
            ratio = logratio.exp()

            mb_advantages = adv[mb_inds]
            if cfg.algorithm.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )
            print(mb_advantages, ratio)
            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - cfg.algorithm.clip_coef, 1 + cfg.algorithm.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if cfg.algorithm.clip_vloss:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -cfg.algorithm.clip_coef,
                    cfg.algorithm.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = (
                pg_loss
                - cfg.algorithm.ent_coef * entropy_loss
                + v_loss * cfg.algorithm.vf_coef
            )
            print("losses", pg_loss, entropy_loss, v_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.algorithm.max_grad_norm)
            optimizer.step()
    return v_loss.detach().cpu().item(), pg_loss.detach().cpu().item(), entropy_loss.detach().cpu().item()

def extract_masked(batch, idx):
    mask = batch.batch == idx
    m2 = batch.mask[mask] == 1
    res = batch.x[mask].reshape(-1)  # [m2]
    res = res[m2.reshape(-1) == 1]
    #print(torch.stack(torch.where(m2),-1).T)
    # print("data",res.shape,m2.shape,mask.shape,m2.sum())
    return res.reshape(-1)


class MCTS_like:
    def __init__(
        self,
        simulator,
        model,
        c=1.0,
        n_sim=10,
        lmbda=0.1,
        gamma=0.99,
        device=None,
    ):
        super().__init__()

        self.simulator = simulator
        self.device = "cpu" if device is None else device

        self.c = c
        self.n_sim = n_sim
        self.model = model
        self.gamma = gamma
        self.lmbda = lmbda
        print("resetting simulator")
        obs, _ = simulator.reset()
        n_actions = obs.action_mask.sum()
        print("here, now running model")
        ws, vs = self.model([obs])
        self.root = Node(
            data=obs,
            W=extract_masked(ws, 0),
            Value=extract_masked(vs, 0),
            Reward=0,
            done=False,
            children=[None for _ in range(n_actions)],
            parent=None,
            Age=0,
            total_W=extract_masked(ws, 0),
            ac_pos=np.stack(np.where(obs.action_mask == 1), -1),
        )

    def mk_node(self, parent: Node, acpos, raw_index: int, age: int):
        #print("action",acpos[1],"position",acpos[0])
        action = acpos[1]
        position = acpos[0]
        zx_graph = parent.data.state_zx_graph.clone()
        obs, reward, terminated, truncated, _ = self.simulator.step(
            action, position=position, pyzx_state=zx_graph
        )
        print("action",self.simulator.rules_list[action],"position",position,"at depth",parent.depth)
        done = terminated or truncated
        ws, vs = self.model([obs])
        parent.expand(
            obs,
            extract_masked(ws, 0),
            extract_masked(vs, 0),
            raw_index,
            reward,
            done,
            age,
        )
        return (
            terminated,
            truncated,
        )

    def expand_tree(
        self,
        age,
    ):
        chooseables = self.root.get_choosables()
        if len(chooseables)==0:
            return None
        #print("age",max([c[0].Age for c in chooseables]),"current age",age, "choice length",len([c[0].Age for c in chooseables]))
        nodes, acts, total_w, vs = zip(*chooseables)
        total_w = torch.stack(total_w) / (torch.tensor([n.depth for n in nodes]) + 1)
        ds = torch.distributions.Categorical(logits=torch.log_softmax(total_w, -1))
        idx = ds.sample()
        sel = nodes[idx]
        actions = acts[idx]
        acpos = sel.ac_pos[actions]
        logp = ds.log_prob(idx)
        (
            terminated,
            truncated,
        ) = self.mk_node(sel, acpos, actions, age)
        reward = self.root.best_reward()
        return reward, terminated, truncated, actions,idx, torch.max(torch.stack(vs)), logp

    def rollout(self, n_steps):
        # first step always has zero reward
        rewards = []
        dones = []
        actions = []
        values = []
        logps = []
        indices = []
        for i in range(1,n_steps + 2):
            res = self.expand_tree(i)
            if res is None:
                break
            r, t1, t2, a,idx, v, logp = res
            rewards.append(r)
            dones.append(t1 or t2)
            actions.append(a)
            values.append(v)
            logps.append(logp)
            indices.append(idx)
        return (
            torch.tensor(rewards)[:-1],
            torch.tensor(values),
            torch.tensor(dones),
            torch.tensor(logps)[:-1],
            torch.tensor(actions)[:-1],
            torch.tensor(indices)[:-1]
        )

    def extract_data(self, rewards, values, dones, logprobs, actions,indices):
        # just use the best reward
        #rewards = torch.zeros_like(rewards)
        #rewards[-1] = self.root.best_reward()
        returns, adv = bootstrap_reward(rewards, values, dones, self.gamma, self.lmbda)
        # get "states"
        return self.root, logprobs, actions, indices, adv, returns, values[:-1],rewards


if __name__ == "__main__":
    from zx_env import zx_env
    from ppo import make_env

    torch.manual_seed(42)
    np.random.seed(42)

    device, device2 = "cpu", "cpu"
