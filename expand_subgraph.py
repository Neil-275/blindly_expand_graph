import pickle as pkl
import random
from dotenv import load_dotenv
import prompt_list
import torch
from utils import extract_numbers, extract_strings, extract_notations, run_llm, extract_answer,fuse_mean, fuse_rrf
from typing import Literal
import numpy as np
load_dotenv()
from queue import Queue
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util


# with open("knowledge_graph/KG_data/FB15k-237-betae/train.txt", "r") as f:
#     train_graph = f.readlines()






def convert_relation(rel: str):
    tmp = rel.split("/")
    if tmp[0] == "+":
        return "/".join(tmp[1:])
    elif tmp[0] == "-":
        return "/".join(tmp[1:][::-1]) 



class ExpandSubgraph:

    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')


    with open("knowledge_graph/KG_data/FB15k-237-betae/id2ent.pkl", "rb") as f:
        id2ent = pkl.load(f)
    with open("knowledge_graph/KG_data/FB15k-237-betae/id2rel.pkl", "rb") as f:
        id2rel = pkl.load(f)
    with open("knowledge_graph/KG_data/FB15k-237-betae/ent2id.pkl", "rb") as f:
        ent2id = pkl.load(f)
    with open("knowledge_graph/KG_data/FB15k-237-betae/FB15k_mid2name.txt", "r", encoding='utf-8') as f:
        ent2name = {}
        for line in f:
            if line.strip():  # Skip empty lines
                mid, name = line.strip().split("\t", 1)
                ent2name[mid] = name


    name2ent = {v:k for k,v in ent2name.items()}  
    id2rel = {k:convert_relation(v) for k,v in id2rel.items()}
    rel2id = {v:k for k,v in id2rel.items()}  

    adj = None
    rel_embs = None
    def __init__(self, n_ent: int, n_rel: int, homoEdges: list, edge_index: list[list], args=None,
                  fuse_func=fuse_mean,
                  use_sub_objectives_a=False,
                  use_sub_objectives_b=False,
                  GoG_simulation=False,
                  GoG_args = None
                  ):
        # not using homoEdges currently
        self.args = args
        self.model = ExpandSubgraph.model.to(args.device)
        self.util = util
        self.k_rel_org = args.k_rel
        self.k_cands_org = args.k_cands
        self.k_rel = args.k_rel
        self.k_cands = args.k_cands
        self.k_decay = 0.9
        self.cands_lim = args.cands_lim
        self.GoG_simulation = GoG_simulation
        self.GoG_args = GoG_args
        if GoG_simulation:
            assert(GoG_args is not None), "Please provide GoG_args when GoG_simulation is True"
        self.orignal_edge_index = np.array(edge_index)
        self.edge_index = self.orignal_edge_index.copy()
        if ExpandSubgraph.adj is None:
            ExpandSubgraph.adj = self.build_adjacency_list()
        if ExpandSubgraph.rel_embs is None:
            ExpandSubgraph.rel_embs = self.model.encode(list(ExpandSubgraph.id2rel.values()), convert_to_tensor=True).to(self.args.device)
        self.query = None
        self.fuse_func = fuse_func
        self.subgraph_key = None
        self.answers_id = None
        self._name_cache = {}
        self.visited = set()
        
        if use_sub_objectives_a or use_sub_objectives_b:
            self.fuse_func = fuse_rrf
        self.use_sub_objectives_a = use_sub_objectives_a
        self.use_sub_objectives_b = use_sub_objectives_b
        self.sub_objectives = None

    def extract_subobjectives(self, query):
        if self.use_sub_objectives_a:
            subobjective_prompt_filled = prompt_list.subobjective_prompt + query[0] + "\nOutput: "
        elif self.use_sub_objectives_b:
            subobjective_prompt_filled = prompt_list.subobjective_prompt2 + query[0] + "\nOutput: "

        # print("Extracting subobjectives with prompt:\n", subobjective_prompt_filled)
        subobjectives  = run_llm(subobjective_prompt_filled, sub_objective = True)
        return subobjectives.res

    def assign_query(self, query):
        self.query = query
        self.query_emb = self.model.encode(self.query['natural_query'], convert_to_tensor=True).to(self.args.device)
        nums = extract_numbers(query['raw_query'])
        notations = extract_notations(query['query_type'])
        
        self.start_entities = []
        for i in range(len(nums)):
            if notations[i] == 'e':
                self.start_entities.append(nums[i])
        self.edge_index = self.orignal_edge_index.copy()
        if self.GoG_simulation:
            # print("Simulating GoG by removing edges...")
            self.remove1hopEdges()
            self.removeGoldRelationPath()
            self.updateEdges(self.edge_index)
        

    def build_adjacency_list(self):
        """
        Currently separating direct and inverse edges.
        Optimized to use defaultdict for better performance.
        """
        # print("Building adjacency list...")
        from collections import defaultdict
        adj = defaultdict(list)

        for i, edges in enumerate(self.edge_index):
            head, rel, tail = edges
            adj[head].append((rel, tail))
        
        return adj

    def id2name(self, idx):
        """Cached version to avoid repeated dictionary lookups."""
        idx = idx.item()
        if idx not in self._name_cache:
            self._name_cache[idx] = ExpandSubgraph.ent2name[ExpandSubgraph.id2ent[idx]]
        return self._name_cache[idx]

    def compare_rel_query_and_return_topk(self, triplets: list,  fuse_func=fuse_mean, ):
        triplets = np.array(triplets)
        unique_rel_ids = np.unique(triplets[:,1])
        rels_emb = ExpandSubgraph.rel_embs[unique_rel_ids]
        scores = self.util.dot_score(self.query_emb, rels_emb).to('cpu')
        top_k_indices = self.return_top_k(scores, fuse_func=fuse_func)
        selected_rels = unique_rel_ids[top_k_indices]

        rel_order = {unique_rel_ids[idx]: rank for rank, idx in enumerate(top_k_indices)}
        filter_mask = (triplets[:, 1] == selected_rels[:, None]).any(axis=0)
        filtered_triplets = triplets[filter_mask]
        
        # Sort by the ranking from top-k
        sorted_keys = np.array([rel_order.get(item[1], float('inf')) for item in filtered_triplets])

        sorted_triplets = filtered_triplets[np.argsort(sorted_keys)]
        # print("sorted_triplets:", sorted_triplets)
        return filtered_triplets


    def return_top_k(self, scores, fuse_func=fuse_mean):
        scores = fuse_func(scores)
        k = min(self.k_rel, scores.shape[1])
        scores, sorted_indices = torch.topk(scores, k, largest=True, dim=-1)
        # print("scores:", scores, "sorted_indices:", sorted_indices)
        return sorted_indices[0].tolist()

    def reset(self):
        self.subgraph_key = None
        self.visited = set()
        self.k_rels = self.k_rel_org
        self.k_cands = self.k_cands_org
        self._name_cache = {}

    def sampleSubgraph(self, query=None):
        assert (self.query is not None or query is not None), "Please assign a query first using assign_query()"
        # print("Sampling subgraph...")
        if query is not None:
            self.assign_query(query)
        adjacency_list = ExpandSubgraph.adj
        start_entities = list(set(self.start_entities.copy()))
        self.reset()
        
        while len(start_entities) > 0:
            # print("hahaha")
            
            triplets = []
            # Fix: Create a copy to avoid modifying list during iteration
            start_entities_copy = start_entities.copy()
            for head_id in start_entities_copy:
                if head_id not in adjacency_list.keys():
                    start_entities.remove(head_id)
                    continue
                
                edges = [(head_id, rel_id, tail_id) for rel_id, tail_id in adjacency_list[head_id]]
                triplets.extend(edges)

            triplets = [triplet for triplet in triplets if 
                        (triplet[0], triplet[1], triplet[2]) not in self.visited]
            if triplets == []:
                break
            # np_triplets is an np.array
            np_triplets = self.compare_rel_query_and_return_topk(
                triplets, fuse_func=self.fuse_func)
            if len(np_triplets) == 0:
                break

            # randomly prune cands
            if len(np_triplets) > self.k_cands:
                sampling_idx = np.random.choice(np.arange(len(np_triplets)), size=self.k_cands, replace=False)
                np_triplets = np_triplets[sampling_idx]

            for triplet in np_triplets:
                self.visited.add((triplet[0], triplet[1], triplet[2]))

                self.visited.add((triplet[2], triplet[1] + -1**(triplet[1] % 2 + 2), triplet[0]))  # Inverse relation
            # print("123")
            # Check if adding these triplets would exceed the candidate limit
            if self.subgraph_key is not None:
                temp_subgraph = np.concatenate([self.subgraph_key, np_triplets], axis=0)
                temp_num_cands = len(np.unique(temp_subgraph[:, [0,2]].flatten()))
                if temp_num_cands > self.cands_lim:
                    # Only add triplets that don't exceed the limit
                    for triplet in np_triplets:
                        temp_subgraph_single = np.concatenate([self.subgraph_key, [triplet]], axis=0)
                        temp_num_cands_single = len(np.unique(temp_subgraph_single[:, [0,2]].flatten()))
                        if temp_num_cands_single <= self.cands_lim:
                            self.subgraph_key = temp_subgraph_single
                        else:
                            break
                    break
                else:
                    self.subgraph_key = temp_subgraph
            else:
                self.subgraph_key = np_triplets
                # print("Initial subgraph_key set.")
            
            start_entities = list({triplet[2] for triplet in np_triplets})
            
            num_cands = len(np.unique(self.subgraph_key[:, [0,2]].flatten()))
            if num_cands >= self.cands_lim:
                break
            
            # self.k_rel = max(1, int(self.k_rel * self.k_decay))
            # self.k_cands = max(1, int(self.k_cands * self.k_decay))
        
        # print(self.subgraph_key)
        if self.subgraph_key is None:
            # If no subgraph was found, return empty arrays/tensors
            empty_nodes = torch.tensor([], dtype=torch.long)
            empty_index = torch.zeros(len(ExpandSubgraph.id2ent), dtype=torch.long)
            empty_edges = torch.tensor([], dtype=torch.long).reshape(0, 3)
            return empty_nodes, empty_index, empty_edges
        
        topk_nodes = np.unique(self.subgraph_key[:, [0,2]].flatten())
        topk_nodes = torch.from_numpy(topk_nodes)
        # topk_nodes = list(topk_nodes)
        node_index = torch.zeros(len(ExpandSubgraph.id2ent), dtype=torch.long)
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
        
        subgraph_key = torch.tensor(self.subgraph_key, dtype=torch.long)
        ## Return 3 tensors
        return topk_nodes, node_index, subgraph_key

    def sampleSubgraphBFS(self, query=None, max_depth: int = None):
        if query is not None:
            self.assign_query(query)
        if not max_depth:
            max_depth = self.args.depth
        # print(max_depth)
        adjacency_list = ExpandSubgraph.adj
        self.reset()
        # print("Starting BFS subgraph sampling...")
        queue = Queue()
        for entity in self.start_entities:
            queue.put((entity, 0))
        subgraph_edges = []

        head = 0
        while head < queue.qsize():
            current_entity, depth = queue.get()
            head += 1

            if depth >= max_depth:
                continue
            
            triplets = [(current_entity, rel_id, tail_id) for rel_id, tail_id in adjacency_list[current_entity]]
            triplets = np.array([triplet for triplet in triplets if (triplet[0], triplet[1], triplet[2]) not in self.visited and self.id2name(triplet[2]) != "UnName_Entity"])
            for triplet in triplets:
                self.visited.add((triplet[0], triplet[1], triplet[2]))

                self.visited.add((triplet[2], triplet[1] + -1**(triplet[1] % 2 + 2), triplet[0])) 
                subgraph_edges.append(triplet)
                queue.put((triplet[2], depth + 1))
        
        if not subgraph_edges:
            # If no subgraph was found, return empty structures
            empty_nodes = np.array([], dtype=np.int64)
            empty_index = torch.zeros(len(ExpandSubgraph.id2ent), dtype=torch.long)
            empty_edges_arr = np.array([], dtype=np.int64).reshape(0, 3)
            return empty_nodes, empty_index, empty_edges_arr

        # Format the output to match sampleSubgraph
        self.subgraph_key = np.array(subgraph_edges, dtype=np.int64)
        topk_nodes = torch.from_numpy(self.subgraph_key[:, [0, 2]].flatten())
        
        node_index = torch.zeros(len(ExpandSubgraph.id2ent), dtype=torch.long)
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
        subgraph_key = torch.tensor(self.subgraph_key, dtype=torch.long)

        return topk_nodes, node_index, subgraph_key


    def evaluate_subgraph(self, type_eval="train"):
        if not self.subgraph_key:
            self.sampleSubgraph()

        subgraph_key = self.subgraph_key
        # print("subgraph has", len(subgraph_key), "triplets.")

        entities_id = set()
        rels = set()
        for h, r, t in subgraph_key:
            entities_id.add(h)
            entities_id.add(t)
            rels.add(r)
        # print("subgraph has", len(entities_id), "unique entities and", len(rels), "relation types.")
        entity_score = self.evaluate_ans_coverage(entities_id, type_eval=type_eval)
        # relation_score = self.evaluate_rel_coverage(rels, type_eval=type_eval)
        return entity_score

    def evaluate_ans_coverage(self, entities_id, type_eval="train"):
        answers_id = set(self.query['answers'])
        entity_score = len(answers_id.intersection(entities_id)) / len(answers_id)

        return entity_score

    def evaluate_rel_coverage(self, rels, type_eval="train"):
        # print(self.raw_query)
        notations = extract_notations(self.query['query_type'])
        rels_ans = extract_numbers(self.raw_query)
        rels_ans = set([rel for i, rel in enumerate(rels_ans) if notations[i] == 'r'])

        relation_score = len(rels_ans.intersection(rels)) / len(rels_ans)
        return  relation_score
    
    
    def getOneSubgraph(self):
        assert (self.query is not None), "Please assign a query first using assign_query()"
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph()
        return [self.start_entities, topk_nodes, node_index, sampled_edges]

    def getBatchSubgraph(self, subgraph_list: list):  
        # print("Getting batch subgraph...")
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []

        for batch_idx in range(batchsize):       
            sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
            num_nodes = len(topk_nodes)
            ent_delta = sum(ent_delta_values) # tính offset.

            # Adding ent_delta to make node indices unique in the batch
            sampled_edges[:,0] = node_index[sampled_edges[:,0]] + ent_delta
            sampled_edges[:,2] = node_index[sampled_edges[:,2]] + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs.append(int(node_index[sub]) + ent_delta)
        
        # [n_batch_ent]
        batch_idxs = torch.LongTensor(batch_idxs)
        # [n_batch_ent]
        abs_idxs = torch.LongTensor(abs_idxs)
        # [n_batch_edges, 3]
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        # [n_batch_edges]
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        # [n_batch]
        query_sub_idxs = torch.LongTensor(query_sub_idxs)    
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges
    
    def updateEdges(self, new_edges):
        self.edge_index = np.array(new_edges)
        ExpandSubgraph.adj = None
        if ExpandSubgraph.adj is None:
            ExpandSubgraph.adj = self.build_adjacency_list()

    def removeGoldRelationPath(self):
        # print("Removing gold relation path edges...")
        gd_relations = []
        for idx, num in zip(extract_notations(self.query['query_type']), extract_numbers(self.query['raw_query'])):
            if idx == 'r':
                gd_relations.append(num)
        answer_set = self.query['answers_id']
        # print(len(gd_relations))
        paths = []
        for entity in self.start_entities:
            paths.append([entity])
        gold_relation_paths = []
        while True:
            # print(123)
            new_paths = []
            for path in paths:
                current_entity = path[-1]
                cur_ent_idx = len(path) - 1
                cur_rel_idx = cur_ent_idx 
                
                if current_entity in answer_set and len(path) == len(gd_relations) + 1:
                    # print("path:", path)
                    gold_relation_paths.append(path)
                    continue
                
                # Check if we've used all relations in the path
                if cur_rel_idx >= len(gd_relations):
                    continue
                    
                dir_rel = gd_relations[cur_rel_idx]
                # Only follow directed edges: current_entity must be the head (column 0)
                match = (self.edge_index[:,1] == dir_rel) & (self.edge_index[:,0] == current_entity)
                tails = np.unique(self.edge_index[match, 2]).tolist()

                # print(tails)
                # print("**************\n")
                for tail in tails:
                    if tail in path:
                        continue
                    # print(tail, path, tail in path)
                    p = path.copy()
                    p.append(tail)
                    new_paths.append(p)
            if len(new_paths) == 0:
                break
            paths = new_paths
            
            # break
        
        gold_triplets = []
        for path in gold_relation_paths:
            # if (len(path) > len(gd_relations) + 1): continue
            tmp = []
            for i in range(len(gd_relations)):
                # print(i)
                triplets = np.array([path[i], gd_relations[i], path[i+1]])
                gold_triplets.append(triplets)
                tmp.append(triplets)
            
        gold_triplets = np.unique(np.array(gold_triplets), axis=0).tolist()

        random.shuffle(gold_triplets)
        num_to_remove = int((self.GoG_args['drop_ratio']) * len(gold_triplets))
        triplets_to_remove = gold_triplets[:num_to_remove]
        
        # Use mask-based removal to avoid index shifting issues
        mask = np.ones(len(self.edge_index), dtype=bool)
        
        for triplet in triplets_to_remove:
            # Find and mark forward edge for removal
            
            matches = np.where((self.edge_index == triplet).all(axis=1))[0]
            # print(triplet, matches)
            if len(matches) > 0:
                mask[matches[0]] = False
            
            # Find and mark inverse edge for removal
            rel_idx = triplet[1] + (-1)**(triplet[1] % 2 + 2)
            inv_triplet = np.array([triplet[2], rel_idx, triplet[0]])
            inv_matches = np.where((self.edge_index == inv_triplet).all(axis=1))[0]
            if len(inv_matches) > 0:
                mask[inv_matches[0]] = False
        
        # Apply mask once to remove all edges
        self.edge_index = self.edge_index[mask]

    def remove1hopEdges(self):
        # print("Removing 1-hop edges between start entities and answers...")
        # print(type(self.edge_index))
        ents = np.array([[a[0], a[2]] for a in self.edge_index])
        start_entities = self.start_entities
        # Use correct key: try 'answers_id' first, fallback to 'answers'
        answer_ids = self.query.get('answers_id', self.query.get('answers', []))
        
        # Build list of 1-hop connections to remove
        gd_doublets = []
        for start_entity in start_entities:
            for answer_id in answer_ids:
                gd_doublets.append(np.array([start_entity, answer_id]))
        
        # Use mask-based removal to avoid index shifting
        mask = np.ones(len(self.edge_index), dtype=bool)
        for doublet in gd_doublets:
            matches = np.where((ents == doublet).all(axis=1))[0]
            mask[matches] = False
        
        self.edge_index = self.edge_index[mask]
            