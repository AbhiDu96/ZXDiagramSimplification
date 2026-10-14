import torch
import torch_geometric.transforms
import torch_geometric.data
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric import nn as gnn
import hydra
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pytorch_metric_learning import losses
from torch.utils.tensorboard import SummaryWriter
import time
import torch.multiprocessing
import os

torch.multiprocessing.set_sharing_strategy("file_system")
from torch_scatter import scatter
import pickle


class ZXGraphDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.picks = 2
        #self.cache = deque(maxlen=256)
        #self.graphs = deque(maxlen=256)
        # self.new_obs = gather_data.remote(self.cfg)

    def __getitem__(self, index):
        with open(f"dataset/item_{index}.pkl","rb") as reader:
            obses = pickle.load(reader)
        idxs = np.random.choice(len(obses), size=(self.picks,), replace=False)
        return [obses[i] for i in idxs]

    def __len__(self):
        # 32 choose 2
        return len(os.listdir("dataset/"))


def worker_init_fun(worker_id):
    # dct=torch.utils.data.get_worker_info()
    # ds = dct.dataset
    # seed=torch.initial_seed()
    np.random.seed(worker_id)


def collate(batch):
    obs = list(zip(*batch))
    out = []
    out += obs[0]
    out += obs[1]
    return torch_geometric.data.Batch.from_data_list(out)


def no_identity_shuffle(items):
    choices = np.arange(items)
    for _ in range(50):
        if np.all(choices != np.arange(items)):
            break
    return choices


class GlobalAggregator(nn.Module):
    def __init__(self, node_size, edge_size, output_size, hidden_channels, n_layers=5):
        super().__init__()
        # first initialize the graph model
        self.node_to_hidden = nn.Sequential(
            nn.BatchNorm1d(node_size),
            nn.Linear(node_size, hidden_channels * 2),
            nn.GLU(),
        )
        self.edge_conv = nn.ModuleList(
            [
                gnn.GATConv(hidden_channels, hidden_channels, edge_dim=edge_size)
                for _ in range(n_layers)
            ]
        )
        self.edge_transform = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(edge_size), nn.Linear(edge_size, edge_size)
                )
                for _ in range(n_layers)
            ]
        )
        # then add the virtual global node
        self.virtual_nodes = torch_geometric.transforms.VirtualNode()
        # output projection
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_channels, output_size), nn.LeakyReLU()
        )
        self.scalar = nn.Parameter(torch.zeros(n_layers))
        self.expander = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(1024),
            #nn.Linear(1024, 1024),
            #nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4096),
        )

    def give_features(self, data: torch_geometric.data.Batch):
        t0 = time.time()
        data = self.virtual_nodes(data)
        nodes = self.node_to_hidden(data.x)
        edges = data.edge_attr.unsqueeze(-1)
        t1 = time.time()
        for idx, (c, t) in enumerate(zip(self.edge_conv, self.edge_transform)):
            nodes = (
                self.scalar[idx] * F.leaky_relu(c(nodes, data.edge_index, edges))
                + nodes
            )
            edges = F.leaky_relu(t(edges)) + edges
        t2 = time.time()
        batch_index = data.batch
        # global_nodes = []
        global_nodes = scatter(nodes, batch_index, dim=0, reduce="mean")
        # for i in range(max(batch_index) + 1):
        #    global_nodes.append(nodes[batch_index == i][-1])
        t3 = time.time()
        # global_nodes = torch.stack(global_nodes)
        print("prepro", t1 - t0, "message passing", t2 - t1, "gathering", t3 - t2)
        return self.output_linear(global_nodes)

    def forward(self, data: torch_geometric.data.Batch):
        output = self.give_features(data)
        return self.expander(output)


def load(path):
    net = torch.load(path, map_location="mps")
    return net


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    folder = f"runs_pretraining/{int(time.time())}_test"
    writer = SummaryWriter(folder)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg, resolve=True).items()])),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "mps" if torch.backends.mps.is_available() and cfg.device == "mps" else "cpu"
    )
    model = GlobalAggregator(2, 1, 64, 256, n_layers=6).to(device)
    #model = load("runs_pretraining/1722929332_test/229000_model.pth")
    loss_fn = losses.VICRegLoss()
    opt = torch.optim.AdamW(model.parameters(), 3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    dataloader = DataLoader(
        ZXGraphDataset(cfg),
        batch_size=256,
        collate_fn=collate,
        num_workers=16,
        prefetch_factor=16,
        persistent_workers=True,
        worker_init_fn=worker_init_fun,
    )
    global_step = 0
    for epoch in range(cfg.epochs):
        print("epoch", epoch)
        for stp, obs in enumerate(dataloader):
            global_step += 1
            # bso = torch_geometric.data.Batch.from_data_list(obs).to(device)
            # o1 = obs[0].to(device,non_blocking=True)
            # o2 = obs[1].to(device,non_blocking=True)
            # es1 = model(o1)
            # es2 = model(o2)
            es1, es2 = torch.chunk(model(obs.to(device)), 2)
            # labels = torch.cat([torch.ones(len(o))*i for i,o in enumerate(os)])
            # hard_tuples = miner(es,labels)
            loss_triplet = loss_fn(es1, ref_emb=es2)
            # regularizer_loss = (1-torch.norm(es,dim=-1)).abs().mean()
            loss = loss_triplet  # +0.001*regularizer_loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            writer.add_scalar(
                "losses/loss", loss_triplet.detach().cpu().item(), global_step
            )
            # "regularizer",regularizer_loss,
            print("losses", loss_triplet, torch.mean((es1 - es2) ** 2), torch.std(es1))
            if global_step > 5_000_000:
                break
            if global_step % 1000 == 0:
                torch.save(model, folder + f"/{global_step}_model.pth")
        if global_step > 5_000_000:
            break
        sched.step()
        writer.add_scalar("losses/lr_schedular", np.mean(sched.get_last_lr()), global_step)

if __name__ == "__main__":
    main()
