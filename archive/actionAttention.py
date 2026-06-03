import torch
from torch import nn
from torch.nn import functional as F


def attention_mask(available_actions:torch.Tensor,neighborhood_sizes:torch.Tensor,n_neighbors:int):
    # available actions is a binary mask containing which actions are choseable
    B, n_nodes, n_actions = available_actions.shape
    # neighborhood_sizes is simply the number N before padding starts
    B, n_nodes = neighborhood_sizes.shape
    mask = torch.ones((B,n_actions,n_neighbors))*(-torch.inf)
    print(mask.shape)
    for i in range(B):
        for j in range(n_nodes):
            mask[i,available_actions[i,j]>=0,:neighborhood_sizes[i,j]] = 0
    return mask


class ActionCrossAttention(nn.Module):
    def __init__(self,action_dim,n_actions, hidden_dim, n_heads):
        super().__init__()
        self.n_actions = n_actions
        self.n_heads = n_heads
        # the "0" entry is for padding
        self.actionspace = nn.Embedding(self.n_actions+1,action_dim,padding_idx=0)
        self.phase_embedder = nn.Linear(1,action_dim)
        #self.node_conversion = nn.Linear(cfg.model.node_dim, cfg.model.action_dim)
        self.kv_embedding = nn.Linear(action_dim,2*hidden_dim)
        self.q_embeddings = nn.Linear(action_dim,hidden_dim)
        self.mhsa = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.out_projection = nn.Sequential(nn.Linear(hidden_dim,2*hidden_dim),nn.LeakyReLU(),
                                            nn.Linear(2*hidden_dim,action_dim))
        self.action_projection = nn.Linear(action_dim,1)

    def step(self,current_state,neighborhood_idx,available_actions, neighborhood_sizes):
        # Get actions and nodes into the same space
        # The neighborhood consists of (B, N_neighbors, n_actions, n_action_dim)
        # therefore we first contract over the actions to get a singelton "node state"
        B, n_nodes,n_neighbors = neighborhood_idx.shape
        B, n_nodes, n_actions,_=current_state.shape
        neighborhood = current_state[torch.arange(B)[:,None,None],neighborhood_idx]
        #print("neighborhood",neighborhood.shape)
        k,v = torch.chunk(self.kv_embedding(neighborhood.mean(-2)),2,-1)
        k = k.reshape(B*n_nodes,n_neighbors,-1)
        v = v.reshape(B*n_nodes,n_neighbors,-1)
        q = self.q_embeddings(current_state).reshape(B*n_nodes,n_actions,-1)
        #print("computed q,k,v. Attending",q.shape,k.shape,v.shape)
        # create the attention mask
        mask = attention_mask(available_actions,neighborhood_sizes,n_neighbors).repeat(self.n_heads*n_nodes,1,1)
        attended,_ = self.mhsa(q,k,v,attn_mask=mask,need_weights=False)
        #print("Attention done, now projecting out",attended.shape)
        # now map back into action_dim
        out = F.leaky_relu(self.out_projection(attended)).reshape(B,n_nodes,n_actions,-1)
        return out
    
    def forward(self,available_actions,neighborhood_sizes, neighborhood_idx, phases, n_iters=10):
        # initialize_state. Unavailable actions are notated as -1, 
        # to account for this, we shift the action indices by 1 to get index==0->padding token
        state = self.actionspace(available_actions+1) + self.phase_embedder(phases).unsqueeze(1)
        print("state",state.shape)
        # state.shape == (B, n_nodes, n_actions,action_dim)
        for _ in range(n_iters):
            state = self.step(state,neighborhood_idx,available_actions,neighborhood_sizes)+state
        return state
    
    def get_action_node(self,available_actions,neighborhood_sizes, neighborhood_idx, phases, graph_sizes,n_iters=10):
        state = self.forward(available_actions,neighborhood_sizes, neighborhood_idx, phases,n_iters)
        action = self.action_projection(state)
        (B,n_nodes,n_actions,_) = action.shape
        mask = torch.ones_like(action)*(-torch.inf)
        print("maskshape",mask.shape,available_actions.shape)
        for i in range(B):
            for j in range(graph_sizes[i]):
                mask[i,:graph_sizes[i],available_actions[i,j]>=0] = 0

        action = action.reshape(B,-1)
        mask = mask.reshape(B,-1)
        
        action_logits = torch.log_softmax(action+mask,-1)
        # now we sample from the resulting action space
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        # this is now "overflattened", meaning we need to convert into indices
        idx = torch.unravel_index(action ,(n_nodes, n_actions))
        # the first number is the node, the second one is the index of the action
        action = torch.stack(idx,-1)
        chosen_states = state[torch.arange(B),idx[0],idx[1]].unsqueeze(1)
        assert torch.allclose(chosen_states[0], state[0,idx[0][0],idx[1][0]])
        print(chosen_states.shape,state.mean(-2).shape)
        # retrieve based on which node has the most similar average state
        similarity = F.cosine_similarity(state.mean(-2),chosen_states)
        return action, action_logits, similarity




if __name__ == "__main__":
    action_dim = 64
    n_actions = 100
    hidden_dim = 64
    n_heads = 4
    model = ActionCrossAttention(action_dim,n_actions,
                                 hidden_dim, n_heads)
    B = 4
    n_nodes = 1024
    graph_sizes = torch.randint(128,n_nodes,size=(B,))
    available_actions = torch.randint(-1,n_actions, (B,n_nodes,20)).sort(descending=True)[0]
    available_actions = available_actions[:,:,(available_actions>=0).sum(1).sum(0)>0]
    print("neighbors shaped",available_actions.shape)
    max_neighbors = 16
    neighbors = torch.randint(-1,n_nodes, (B,n_nodes,max_neighbors,)).sort(descending=True)[0]
    result = model(available_actions, (neighbors>=0).sum(-1), neighbors, torch.rand(n_nodes,1), 4)
    print(result.shape)
    print("Trying to get a hard action")
    hard_action,action_logits,state_sims = model.get_action_node(available_actions, (neighbors>=0).sum(-1), neighbors, torch.rand(n_nodes,1),graph_sizes, 4)
    print("hard action",hard_action)
    print("the first action would be: execute", available_actions[0,hard_action[0,0],hard_action[0,1]], "on node number",hard_action[0,0])
    print("action_logits",action_logits)
    assert available_actions[0,hard_action[0,0],hard_action[0,1]] >=0
    B = 3
    n_nodes = 32
    available_actions = torch.randint(-1,n_actions, (B,n_nodes,3))
    max_neighbors = 32
    neighbors = torch.randint(-1,n_nodes, (B,n_nodes,max_neighbors,)).sort(descending=True)[0]
    result = model(available_actions, (neighbors>=0).sum(-1), neighbors,torch.rand(n_nodes,1), 4)
    


        