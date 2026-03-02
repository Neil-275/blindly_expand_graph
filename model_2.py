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
        # self.pre_ln = nn.LayerNorm(in_dim)
        for _ in range(n_layers):
            current_dim = in_dim
            layers = []    
            for h_dim in hidden_dims:
                layers.append(nn.LayerNorm(current_dim))
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
    def __init__(self, in_dim, out_dim, attn_dim, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*237, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, gnn_emb_rel=None, mapping=None, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]

        if gnn_emb_rel is not None:
            hr = gnn_emb_rel[mapping[rel]]
            # Check if q_rel is already an embedding (tensor with >1 dimension) or an ID
            if len(q_rel.shape) > 1:  # q_rel is already an embedding
                h_qr = q_rel.squeeze(0) if q_rel.dim() > 1 else q_rel  # handle batch dimension
            else:  # q_rel is an ID, lookup embedding
                h_qr = gnn_emb_rel[mapping[q_rel]][r_idx if r_idx is not None else 0]
        else: # not using llm_emb, use original relation embedding
            hr = self.rela_embed(rel) # relation embedding of each edge
            # Check if q_rel is already an embedding or an ID
            if len(q_rel.shape) > 1:  # q_rel is already an embedding
                h_qr = q_rel.squeeze(0) if q_rel.dim() > 1 else q_rel  # handle batch dimension
            else:  # q_rel is an ID, lookup embedding
                h_qr = self.rela_embed(q_rel)[r_idx if r_idx is not None else 0] # use batch_idx to get the query relation
        
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
    def __init__(self, params, loader, mode="train"):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.loader = loader
        if hasattr(params, 'llm_emb'):
            self.llm_emb = params.llm_emb
        self.n_ent = params.n_ent
        self.n_rel = params.n_rel
        act = nn.ReLU()
        self.active_layers = getattr(params, 'active_layer', self.n_layer) 
        # self.query_rela_embed = nn.Embedding(2*237+1, self.hidden_dim)
       

        if self.params.initializer == 'relation': 
            self.query_rela_embed = nn.Embedding(2*self.n_rel, self.hidden_dim)
        

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        

    def assign_rel_emb(self, embeddings):
        self.llm_emb = embeddings
        self.llm_emb = self.llm_emb.to(torch.device("cuda:0"))

    def forward(self, q_sub, q_rel, subgraph_data, projectors, mode='train', use_llm = False, return_hidden=False):
        ''' forward with extra propagation '''
        n = len(q_sub) # number of queries in the batch
        batch_idxs, abs_idxs, query_sub_idxs, _, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()

        
        
        # print("batch_sampled_edges shape:", batch_sampled_edges.shape)
        # initialize the hidden
        # random select a version for each unique relation
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            if use_llm:
                if mode == "train":
                    q_ver = torch.randint(0,self.params.train_ver,(1,)).item()
                else:
                    q_ver = torch.randint(self.params.train_ver,self.llm_emb.shape[1],(1,)).item()
                hidden[query_sub_idxs, :] = projectors.models[self.active_layers](self.llm_emb[q_rel, q_ver])
            else:
                hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]


        all_rels = torch.cat([batch_sampled_edges[:,1], q_rel], dim = 0)
        unique_rels = torch.unique(all_rels)
        mapping = torch.zeros(all_rels.shape[0], dtype=torch.long).cuda()
        ### mapping from edge to relation index in unique_rels
        for i, r in enumerate(unique_rels):
            mapping[all_rels == r] = i
        # propagation
        for i in range(self.n_layer): # fix later
            # forward
            if use_llm:
                if mode == "train":
                    q_ver = torch.randint(0,self.params.train_ver,(1,)).cuda()
                else:
                    q_ver = torch.randint(self.params.train_ver,self.llm_emb.shape[1],(1,)).cuda()
                llm_emb_rel = self.llm_emb[unique_rels, q_ver, :]

                gnn_emb_rel = projectors.models[i](llm_emb_rel)
                # gnn_emb_rel_mapped = gnn_emb_rel[mapping.long()]
                hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                            gnn_emb_rel,
                                            mapping,
                                            shortcut=self.params.shortcut)
            else:
                hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
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
        # scores = scores.cpu()
        return scores   
    
    def forward_inference(self, q_sub, query_relation_emb, subgraph_data, projectors):
        n = len(q_sub) # number of queries in the batch (should be 1 for inference)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)

        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()

        # print(hidden[query_sub_idxs,:].shape, query_relation_emb.shape)
        hidden[query_sub_idxs, :] = projectors.models[self.active_layers](query_relation_emb)

        if self.params.concatHidden: hidden_list = [hidden]

        # Create relation mapping for subgraph edges only (no query relation ID available)
        all_rels = batch_sampled_edges[:,1]
        unique_rels = torch.unique(all_rels)
        mapping = torch.zeros(all_rels.shape[0], dtype=torch.long).cuda()
        ### mapping from edge to relation index in unique_rels
        for i, r in enumerate(unique_rels):
            mapping[all_rels == r] = i
        
        # propagation through layers
        for i in range(self.n_layer):
            # For inference, we can use a fixed version or the last version of LLM embeddings
            if hasattr(self, 'llm_emb') and self.llm_emb is not None:
                llm_emb_rel = self.llm_emb[unique_rels, :]
                # print(llm_emb_rel.shape, llm_emb_rel.device, next(projectors.models[i].parameters()).device)
                gnn_emb_rel = projectors.models[i](llm_emb_rel)
                gnn_query_rel_emb = projectors.models[i](query_relation_emb)
                # Modified GNNLayer call for inference - pass query_relation_emb directly
                hidden = self.gnn_layers[i](q_sub, gnn_query_rel_emb, None, hidden, batch_sampled_edges, n_node,
                                            gnn_emb_rel,
                                            mapping,
                                            shortcut=self.params.shortcut)
            else:
                # Use standard relation embeddings - pass query_relation_emb directly
                hidden = self.gnn_layers[i](q_sub, query_relation_emb, None, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # activation signal and GRU processing (same as main forward)
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout (same as main forward)
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            # print("Hidden shape before readout:", hidden.shape)
            # dummy return
            # logits =  torch.randn(hidden.shape[0]).cuda()
            # return logits
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        return scores