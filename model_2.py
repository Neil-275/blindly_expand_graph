#from one-shot-subgraph
import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np

class Projector(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, n_layers, dropout=0.2):
        super(Projector, self).__init__()
        
        self.models = []
        # current_dim = in_dim
        self.n_layers = n_layers
        for _ in range(n_layers):
            current_dim = in_dim
            layers = []    
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                # print(layers[-1].weight.shape)
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_dim = h_dim 
            
            layers.append(nn.Linear(current_dim, out_dim))
            model = nn.Sequential(*layers)
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

    def forward(self, layer_idx, x):
        return self.models[layer_idx](x)
    

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = torch.from_numpy(np.load('LLM_relation_embeddings.npy')).float().cuda()
        # self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        # self.projector = Projector(5120, [1024, 512], in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, projector, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        hr = projector(self.rela_embed[rel]) # relation embedding of each edge
        h_qr = projector(self.rela_embed[q_rel][r_idx]) # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))

        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class GNN_auto(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, subgraph_data, projectors, mode='train', return_hidden=False, score_all=True):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        projectors[i],
                                        shortcut=self.params.shortcut)
            

            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        if not score_all:
            return scores

        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores

        if return_hidden:
            # return final hidden states along with scores
            hidden_all = torch.zeros((n, self.loader.n_ent, self.hidden_dim)).cuda()
            if self.params.concatHidden and self.params.readout == 'linear':
                # if we concatenated hidden states for linear readout, use the original hidden
                final_hidden = hidden_list[-1]  # use the last layer's hidden states
            else:
                final_hidden = hidden
            hidden_all[batch_idxs, abs_idxs] = final_hidden
            return scores_all, hidden_all
        
        return scores_all
    
    def get_node_embeddings(self, q_sub, q_rel, subgraph_data):
        with torch.no_grad():
            _, hidden_all = self.forward(q_sub, q_rel, subgraph_data, return_hidden=True)
        
        batch_idxs, abs_idxs, _, _, _ = subgraph_data
        return hidden_all, batch_idxs, abs_idxs
    
    # def get_score(self, q_sub, q_rel, subgraph_data):
    #     with torch.no_grad():
    #         scores_all = self.forward(q_sub, q_rel, subgraph_data)
    #     return scores_all