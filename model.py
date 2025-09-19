import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as geo
from torch_geometric import nn as gnn
from torch_geometric.data import Batch
from torch.distributions import Categorical
from torch_geometric.utils import scatter
import time


class ActionModel(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.model_type = model_type
        # We want attention between the
        if model_type == "GAT":
            self.gat = gnn.GATConv(meta_size, meta_size, heads=n_heads, concat=False)
        elif model_type == "ActionAtt":
            self.gat = GraphCrossAttention(n_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(action_dim),
            # nn.Linear(hidden_dim, action_dim),
        )
        self.policy_projection = nn.Linear(action_dim, 1)
        self.value_projection = nn.Linear(action_dim, 1, bias=False)
        # we say that the zeroth embedding is the "padding" one
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data, action_mask):
        nodes, n_actions = action_mask.shape
        # we have a one-hot action mask, we first convert this to indices
        # note that the zero index is padding, so we shift all action indices by 1
        # t0=time.time()
        indices = (
            torch.arange(1, action_mask.shape[-1] + 1, device=self.device)[None, :]
            * action_mask
        )
        # this is now our padded "state"
        embs = self.action_embedder(indices).reshape(nodes, -1)
        embs = self.message_passing_loop(embs, action_mask, data)
        embs = self.action_masking(embs, action_mask).reshape(nodes, self.n_actions, -1)
        # print("message pass",time.time()-t0)
        # policy projection, also mask out the wrongs
        # print("action_mask\n",action_mask[:8])
        mask = torch.where(action_mask == 0, -torch.inf, 0)
        # print("mask\n",mask[:8])
        # print(mask[:5])
        policy = self.policy_projection(embs).squeeze(-1)
        # print("policy masked",policy[:5])
        policy = policy + mask
        # print("policy masked",policy[:5])

        value = self.value_projection(embs).squeeze(-1)
        return policy, value

    def action_masking(self, embs, action_mask):
        N = embs.shape[0]
        embs = embs.reshape(N, self.n_actions, -1)
        embs = embs * action_mask[..., None]
        # print("embs",embs[:3],action_mask[:3])
        embs = embs.reshape(N, -1)
        return embs

    def message_passing_loop(self, embs, action_mask, data):
        N = embs.shape[0]
        for i in range(self.n_message_passing):
            embs = self.action_masking(embs, action_mask)
            embs_old = embs
            if self.model_type == "GAT":
                # now we mask out the actions that don't actuall exist (i.e. set them to 0)
                embs = self.gat(embs, data.edge_index)
                # embs = self.action_masking(embs, action_mask)
            elif self.model_type == "ActionAtt":
                embs = embs.reshape(N, self.n_actions, -1)
                embs = self.gat(embs, data.edge_index, action_mask).reshape(N, -1)
            embs = (
                self.mlp(embs.reshape(N, self.n_actions, -1)).reshape(N, -1) + embs_old
            )
        return embs

    def get_action_and_value(self, next_obs, actions=None):
        # t0 = time.time()
        data = Batch.from_data_list([o.Graph for o in next_obs]).to(self.device)
        # print(torch.unique(data.batch))
        action_mask = torch.cat([o.action_mask for o in next_obs]).to(self.device)
        policy, value = self(data, action_mask)
        # print("policy",policy.shape,"value",value.shape)
        # value is the mean of whatever the entire network thinks
        # one could also argue it should be the disjoint sum over all neighborhoods,
        # but that would be computationally insane
        # now split the
        sample_actions = actions is None
        actions = torch.zeros(len(next_obs), 2, device=self.device)
        logprob = torch.zeros(len(next_obs), device=self.device)
        entropy = torch.zeros(len(next_obs), device=self.device)
        values = torch.zeros(len(next_obs), device=self.device)
        # print("value",value.shape)
        for i in range(len(next_obs)):
            # print(policy[data.batch == 0].shape)
            p = policy[data.batch == i]
            nnodes, nacts = p.shape
            p = torch.log_softmax(p.reshape(-1), -1)
            # print(p)
            # print(p[:64])
            probs = Categorical(logits=p)
            if sample_actions:
                samp = probs.sample()
                s = torch.stack(torch.unravel_index(samp, (nnodes, nacts)))
                actions[i] = s
            tmp = actions[i][0] * nacts + actions[i][1]
            logprob[i] = probs.log_prob(tmp)
            entropy[i] = probs.entropy()
            values[i] = value[data.batch == i].mean()
        # print("forward time",time.time()-t0)
        return actions, logprob, entropy, values

    def get_value(self, next_obs):
        return self.get_action_and_value(next_obs)[-1]


def graphnorm(data, batch):
    for i in range(max(batch)):
        data[batch == i] = (data[batch == i] - data[batch == i].mean()) / (
            data[batch == i].std() + 1e-5
        )
    return data

class SlowNorm(nn.Module):
    """
    This slowly normalizes the inputs based on the historical means/stds.
    This is particularly useful to make sure the features are from N(0,1).
    Different from batchnorm, this behaves the same during training and evaluation.
    It also does not have features, and it _always_ uses the historical information.
    This is different from batchnorm which only uses the historical info for evaluation,
    and uses the batch-statistics during training.
    """
    def __init__(self,n_channels,momentum=1e-5):
        super().__init__()
        self.n_channels=n_channels
        self.momentum=momentum
        self.register_buffer("mean",torch.zeros(self.n_channels))
        self.register_buffer("std",5*torch.ones(self.n_channels))
    def forward(self,x):
        if self.training:
            with torch.no_grad():
                if x.shape[0]>1:
                    self.mean = self.mean*(1-self.momentum)+x.mean(0)*self.momentum
                    self.std = self.std*(1-self.momentum)+x.std(0)*self.momentum
        return (x-self.mean)/(self.std+1e-5)


class GraphCrossAttention(nn.Module):
    def __init__(
        self,
        n_actions,
        action_dim,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.queries = nn.Linear(self.action_dim, 1, bias=False)
        self.keys = nn.Linear(self.action_dim, 1, bias=False)
        self.values = nn.Linear(self.action_dim, self.action_dim, bias=False)
        self.aggregator = gnn.SimpleConv()
        self.n_actions = n_actions

    def forward(self, embs, edge_index, action_mask):
        # should have shape (N,n_actions,action_dim)
        weighted = self.attend(embs, edge_index, action_mask)
        return weighted

    def attend(self, embs, edge_index, action_mask):
        # get all
        # average exponential action
        N, n_act, dim = embs.shape
        k = self.keys(embs.mean(-2, keepdim=True))
        q = self.queries(embs)
        v = self.values(embs.mean(-2, keepdim=True))
        # now do a GAT-like
        alpha = torch.exp(F.leaky_relu(k[edge_index[0]] + q[edge_index[1]]))
        # aggregates each action based on the edge weights
        # i.e. map neighbors aggregate to actions, based on the weight of _each action_ to the neighbor
        weighted = torch.vmap(
            lambda a: self.aggregator(v.reshape(N, -1), edge_index, edge_weight=a),
            in_dims=1,
            out_dims=1,
        )(alpha)
        return weighted


class MCTS_like_model(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.model_type = model_type
        # We want attention between the
        if model_type == "GAT":
            self.gat = gnn.GATConv(meta_size, meta_size, heads=n_heads, concat=False)
        elif model_type == "ActionAtt":
            self.gat = GraphCrossAttention(n_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(action_dim),
            # nn.Linear(hidden_dim, action_dim),
        )
        self.weight_projection = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1, bias=False)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LeakyReLU(), nn.Linear(256, 1, bias=False)
        )
        # with torch.no_grad():
        #    self.value_projection.weight.data.zero_()
        #    self.weight_projection.weight.data.zero_()
        # we say that the zeroth embedding is the "padding" one
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, next_obs):
        data = Batch.from_data_list([o.Graph for o in next_obs]).to(self.device)
        # print(torch.unique(data.batch))
        action_mask = torch.cat([o.action_mask for o in next_obs]).to(self.device)
        nodes, n_actions = action_mask.shape
        # we have a one-hot action mask, we first convert this to indices
        # note that the zero index is padding, so we shift all action indices by 1
        # t0=time.time()
        indices = (
            torch.arange(1, action_mask.shape[-1] + 1, device=self.device)[None, :]
            * action_mask
        )
        # this is now our padded "state"
        embs = self.action_embedder(indices).reshape(nodes, -1)
        embs = self.message_passing_loop(embs, action_mask, data).reshape(
            nodes, self.n_actions, -1
        )
        # print("embs mean",embs.mean(),embs.std())
        # embs = self.action_masking(embs, action_mask).reshape(nodes, self.n_actions, -1)
        ws = self.weight_projection(embs).squeeze(-1)
        # print("policy masked",policy[:5])
        # print("policy masked",policy[:5])
        vs = self.value_projection(embs).squeeze(-1)
        # compute mean value over each graph
        for i in range(max(data.batch)):
            mx = vs[data.batch == i].max()
            vs[data.batch == i] = mx
        weights = data.clone()
        weights.x = ws
        weights.mask = action_mask
        values = data.clone()
        values.x = vs
        values.mask = action_mask

        return weights, values

    def action_masking(self, embs, action_mask):
        N = embs.shape[0]
        embs = embs.reshape(N, self.n_actions, -1)
        embs = embs * action_mask[..., None]
        # print("embs",embs[:3],action_mask[:3])
        embs = embs.reshape(N, -1)
        return embs

    def message_passing_loop(self, embs, action_mask, data):
        N = embs.shape[0]
        for i in range(self.n_message_passing):
            # embs = self.action_masking(embs, action_mask)
            embs = graphnorm(embs, data.batch)
            embs_old = embs
            if self.model_type == "GAT":
                # now we mask out the actions that don't actuall exist (i.e. set them to 0)
                embs = self.gat(embs, data.edge_index)
                # embs = self.action_masking(embs, action_mask)
            elif self.model_type == "ActionAtt":
                embs = embs.reshape(N, self.n_actions, -1)
                embs = self.gat(embs, data.edge_index, action_mask).reshape(N, -1)
            embs = (
                self.mlp(embs.reshape(N, self.n_actions, -1)).reshape(N, -1) + embs_old
            )
        return embs

    def to(self, device):
        self.device = device
        super().to(device)


class ActionNet(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.meta_size = meta_size
        self.model_type = model_type
        # We want attention between the
        self.feat =  nn.Sequential(nn.Linear(8, self.hidden_dim), nn.LeakyReLU()),
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()
                ),
            ]
        )
        self.position_projection = nn.Linear(meta_size, 1)
        self.action_projection = nn.Linear(meta_size, self.n_actions)
        self.value_projection = nn.Sequential(
            nn.Linear(action_dim, action_dim), nn.LeakyReLU(), nn.Linear(action_dim, 1)
        )
        self.value_projection[-1].weight.data.zero_()
        self.value_projection[-1].bias.data.zero_()
        # we say that the zeroth embedding is the "padding" one
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        """indices = (
            torch.arange(1, data.action_mask.shape[-1] + 1, device=self.device)[None, :]
            * data.action_mask
        )"""
        # embs = self.action_embedder(indices).reshape(nodes, -1)
        # 1.
        embs = self.feat(data)
        for m in self.mlp:
            embs=m(embs)+embs
        # self.gat(embs, data.edge_index)
        # now get the action likelihods
        pos_unmasked = self.position_projection((embs_out + embs))
        graph = scatter((embs_out + embs), data.batch, reduce="mean").reshape(
            data.batch.max() + 1, -1
        )
        action_unmasked = self.action_projection(graph)
        action_unmasked = action_unmasked[data.batch]
        print("action_unmasked", action_unmasked.shape)

        # mask out
        values = self.value_projection(
            (embs_out + embs).reshape(nodes, n_actions, -1)
        ).reshape(nodes, n_actions)
        return pos_unmasked, action_unmasked, values


class TreeNet(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.model_type = model_type
        # We want attention between the
        self.feat =  nn.Sequential(SlowNorm(8),nn.Linear(8, self.hidden_dim), nn.LeakyReLU())
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 2*self.hidden_dim), nn.GLU(),
                ),
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 2*self.hidden_dim), nn.GLU(),
                ),
            ]
        )

        self.priority_prediction = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim, 1))
        self.value_prediction = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),nn.LeakyReLU(),nn.Linear(hidden_dim, 1))
        # we say that the zeroth embedding is the "padding" one
        self.action_embedder = nn.Embedding(
            self.n_actions + 1, embedding_dim=action_dim, padding_idx=0
        )
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        embs = self.feat(data)
        for m in self.mlp:
            embs=m(embs)+embs
        # now get the action likelihods
        node_prios = self.priority_prediction(embs).reshape(-1)
        node_values = self.value_prediction(embs).reshape(-1)
        return node_prios, node_values


class DummyActionModle(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        meta_size = int(action_dim * self.n_actions)
        self.meta_size = meta_size
        self.model_type = model_type
        self.n_message_passing = n_message_passing
        self.device = device

    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        pos_unmasked = torch.ones(nodes, 1, device=self.device)
        action_unmasked = torch.ones(nodes, n_actions, device=self.device)
        values = torch.zeros(data.action_mask.shape, device=self.device)
        return pos_unmasked, action_unmasked, values


class ActionEmbeddingModel(nn.Module):
    def __init__(self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,):
        super().__init__()
        #self.action_emb = nn.Embedding(1024,hidden_dim-action_dim)
        self.projection = nn.Linear(4,hidden_dim)
        self.transf = nn.ModuleList([gnn.conv.TransformerConv(hidden_dim,hidden_dim,n_heads) for _ in range(n_message_passing)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim*2),nn.LeakyReLU(),nn.Linear(2*hidden_dim,hidden_dim)) for _ in range(n_message_passing)])
        self.value_proj = nn.Linear(hidden_dim,1)
        self.action_proj = nn.Linear(hidden_dim,n_actions)
        self.pos_proj = nn.Linear(hidden_dim,1)
        self.n_message_passing=n_message_passing


    def forward(self, data):
        nodes, n_actions = data.action_mask.shape
        p = self.projection(data.x)
        for i in range(self.n_message_passing):
            p = self.transf[i](p,data.edge_indx)
            p = self.ff[i](p)
        pos_unmasked = self.pos_proj(p)
        values = self.pos_proj(p)
        action_unmasked = self.action_proj(p)
        #pos_unmasked = torch.ones(nodes, 1, device=self.device)
        #action_unmasked = torch.ones(nodes, n_actions, device=self.device)
        #values = torch.zeros(data.action_mask.shape, device=self.device)
        return pos_unmasked, action_unmasked, values

class BundleNet(nn.Module):
    def __init__(
        self,
        action_dim,
        n_actions,
        hidden_dim,
        n_heads,
        n_message_passing,
        device,
        model_type,
    ) -> None:
        super().__init__()
        self.nodenet = DummyActionModle(
            action_dim,
            n_actions,
            hidden_dim,
            n_heads,
            n_message_passing,
            device,
            model_type,
        )
        self.treenet = TreeNet(
            action_dim,
            n_actions,
            hidden_dim,
            n_heads,
            n_message_passing,
            device,
            model_type,
        )
