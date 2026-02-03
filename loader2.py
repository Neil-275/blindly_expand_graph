import os
import torch
import numpy as np
from collections import defaultdict
import pickle as pkl                 
import time
from torch.utils.data import Dataset
import random


def getBatchSubgraph(subgraph_list: list):  
    batchsize = len(subgraph_list)
    ent_delta_values = [0]
    batch_sampled_edges = []
    batch_idxs, abs_idxs = [], []
    query_sub_idxs = []
    query_tail_idxs = []
    edge_batch_idxs = []
    
    for batch_idx in range(batchsize):       
        sub, obj, topk_nodes, _, sampled_edges = subgraph_list[batch_idx]
        num_nodes = len(topk_nodes)
        ent_delta = sum(ent_delta_values) 
        
        # 1. Sort nodes to use searchsorted reliably
        topk_nodes_sorted, sort_indices = torch.sort(topk_nodes)
        
        # 2. Map edges: Global -> Sorted Local -> Absolute Batch Local
        # searchsorted finds the index 'i' such that topk_nodes_sorted[i] == global_id
        src_indices = sampled_edges[:, 0].contiguous()
        dst_indices = sampled_edges[:, 2].contiguous()
        edge_src_local = torch.searchsorted(topk_nodes_sorted, src_indices)
        edge_dst_local = torch.searchsorted(topk_nodes_sorted, dst_indices)
        
        # Apply the offset to move from subgraph-local to batch-local
        sampled_edges[:, 0] = edge_src_local + ent_delta
        sampled_edges[:, 2] = edge_dst_local + ent_delta
        
        batch_sampled_edges.append(sampled_edges)
        edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

        # 3. Map query nodes (Subject and Tail)
        sub_local = torch.searchsorted(topk_nodes_sorted, sub)
        obj_local = torch.searchsorted(topk_nodes_sorted, obj)
        
        query_sub_idxs.append(sub_local + ent_delta)
        query_tail_idxs.append(obj_local + ent_delta)

        # 4. Update Batch Metadata
        ent_delta_values.append(num_nodes)
        batch_idxs += [batch_idx] * num_nodes
        # We store the sorted version so that the local indices [0...N] 
        # correctly correspond to the entries in abs_idxs
        abs_idxs += topk_nodes_sorted.tolist()

    # Convert to Tensors
    batch_idxs = torch.LongTensor(batch_idxs)
    abs_idxs = torch.LongTensor(abs_idxs)
    batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
    edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
    query_sub_idxs = torch.cat(query_sub_idxs).squeeze()
    query_tail_idxs = torch.cat(query_tail_idxs).squeeze()
    # print("query_tail_idxs:", query_tail_idxs)
    tmp = torch.zeros(len(batch_idxs), dtype=torch.bool)
    tmp[query_tail_idxs] = 1
    query_tail_idxs = tmp
    
    return batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs, batch_sampled_edges

class DataLoader2(Dataset):

    def __init__(self, data, mode='train'):
        self.mode = mode
        # self.task_dir = args.data_path
        # print(self.mode)
        self.fine_tuning_data = data
        self.queries = []
        self.map_subgraph = []
        if self.mode == "train":
            for i in range(len(self.fine_tuning_data)):
                self.queries.extend(self.fine_tuning_data[i]['drop_edges'])
                self.map_subgraph.extend([i] * len(self.fine_tuning_data[i]['drop_edges']))
        elif self.mode == "val":
            for i in range(len(self.fine_tuning_data)):
                queries, answers = self.load_query(self.fine_tuning_data[i]['drop_edges'])
                for query, answer in zip(queries, answers):
                    self.queries.append((query, answer)) 
                    self.map_subgraph.append(i)
                    # print(123)

    def __len__(self):  
        return len(self.queries)
    

    def __getitem__(self, index):
        query = self.queries[index]

        if self.mode == "train":
            sub, rel, obj = query
            sub = torch.LongTensor([sub]).unsqueeze(0)
            rel = torch.LongTensor([rel]).unsqueeze(0)
            obj = torch.LongTensor([obj]).unsqueeze(0)
            subgraph_data = self.fine_tuning_data[self.map_subgraph[index]]['subgraph']
        elif self.mode == "val":
            q, ans = query
            sub, rel = q
            sub, rel = torch.LongTensor([sub]), torch.LongTensor([rel])
            obj = torch.LongTensor(ans) #gồm global ids của các tail entities
            subgraph_data = self.fine_tuning_data[self.map_subgraph[index]]['subgraph']

        top_k_nodes, node_index, sampled_edges = subgraph_data
        top_k_nodes = top_k_nodes.clone()
        node_index = node_index.clone()
        sampled_edges = sampled_edges.clone()
        subgraph_data = (sub, obj, top_k_nodes, node_index, sampled_edges)
        return sub, rel, obj, subgraph_data
    
    def collate_fn(self, data):
        subs = torch.stack([_[0] for _ in data], dim=0)
        rels = torch.stack([_[1] for _ in data], dim=0)
        if self.mode == "train":
            objs = torch.stack([_[2] for _ in data], dim=0)
        else:
            objs = torch.cat([_[2] for _ in data], dim=0) # gồm global ids của các tail entities trong batch
        subgraph_list = [_[3] for _ in data]
        batch_subgraph = getBatchSubgraph(subgraph_list)
        
        # NOTE: we can not return sparse tensor here
        # thus, we return its indices and values which are dense tensor.
        batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs, batch_sampled_edges = batch_subgraph
        return subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs, batch_sampled_edges


    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers