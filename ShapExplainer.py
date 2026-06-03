import torch
import numpy as np
import shap
from zx_env import extract_circuit
from zx_env import random_circuit
from TreePolicy import start_tree
from ppo import make_env
import ray
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf, read_write
from models import BundleNet
import os

class DummyExtractor(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model =model
        self.model.eval()
    def forward(self,x):
        x=torch.tensor(x)
        with torch.inference_mode():
            return self.model(x)[0].reshape(-1,1).numpy()



class Exploration(Dataset):
    def __init__(self, trees):
        super().__init__()
        self.trees=trees
        self.items = []
        for t in trees:
            for i in t.infos:
                self.items+=[i["feats"]]
        # deduplicate
        self.items = torch.stack(self.items)
        print("items",self.items.shape)
    def __getitem__(self,idx):
        return self.items[idx]
    def __len__(self):
        return len(self.items)

@ray.remote
def deploy_agents(cfg, state, agent):
    env = make_env(cfg, "cpu")
    start = state.to_graph()
    next_obs = None
    for it in range(cfg.validation.search_loops):
        print("LOOP ROUND", it)
        next_obs, info = env.reset(initial_circuit_graph=start)
        next_obs = start_tree(next_obs.Graph, next_obs.state_zx_graph, info=info)
        for step in range(0, cfg.algorithm.num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                # make a temporary batch from our data:
                effect, total_log, _, value = next_obs.select(agent)

            # TRY NOT TO MODIFY: execute the game and log data.
            new_tree, rew, term, trunc, info = next_obs.expand(effect, env)
            next_done = np.logical_or(term, trunc)
            next_obs, next_done = (
                new_tree,
                torch.Tensor(next_done),
            )
        print(
            "=====" * 5 + "\n",
            f"result of iteration {it}",
            extract_circuit(next_obs.get_best_node())[0].twoqubitcount(),
            "\n" + "=====" * 5,
        )
        start = extract_circuit(next_obs.get_best_node().clone())[0].to_graph()

    return (
        next_obs
    )  #

def get_data(cfg,agent,size_data):
    datas = [random_circuit(
                    n_qubit=cfg.validation.n_qubit,
                    num_gates=cfg.validation.n_gate,
                    p_two_qubit=cfg.validation.p_two_qubit,
                    p_H=cfg.validation.p_H,
                    p_z=cfg.validation.p_z,
                    p_x=cfg.validation.p_x,
                    many_pi_gates=cfg.validation.many_pi_gates,
                    clifford_plus_T=cfg.validation.clifford_plus_T,
                ) for _ in range(size_data)]
    res=[]
    for d in datas:
        res.append(deploy_agents.remote(cfg,d,agent))
    dataset = Exploration(ray.get(res))
    return dataset

def get_shap(agent,dataset):
    dl=DataLoader(dataset,batch_size=2048,shuffle=True)
    ref = next(iter(dl))
    print(ref.shape)
    m=DummyExtractor(agent.treenet)
    print(m(ref))
    exp = shap.KernelExplainer(m,ref.numpy())
    dl=DataLoader(dataset,batch_size=64,shuffle=True)
    dataset = next(iter(dl))
    shap_values = exp(dataset.numpy())
    print(shap_values)
    shap.plots.bar(shap_values)
    plt.savefig("barplot.png")
    shap.plots.beeswarm(shap_values)
    plt.savefig("beeswarmplot.png")



@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    path = cfg.validation.model_path
    weights = torch.load(path)
    env = make_env(cfg, "cpu")
    agent = BundleNet(
        cfg.model.action_dim,
        env.action_space.n,
        hidden_dim=cfg.model.hidden_dim,
        n_heads=cfg.model.n_heads,
        n_message_passing=cfg.model.n_message_passing,
        device="cpu",
        model_type=cfg.model.model_type,
    )
    agent.load_state_dict(weights)
    print("loaded weights")
    dataset = get_data(cfg,agent,64)
    get_shap(agent,dataset)

if __name__ == "__main__":
    ray.init(address=os.environ["ip_head"])
    print("environment",os.environ["ip_head"])
    print("Nodes in the Ray cluster:")
    print(ray.nodes())
    main()