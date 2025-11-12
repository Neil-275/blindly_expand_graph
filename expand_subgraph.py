import pickle as pkl
import random
from dotenv import load_dotenv
import prompt_list
import torch
from utils import extract_numbers, extract_strings, extract_notations
from typing import Literal
import numpy as np
load_dotenv()



# with open("knowledge_graph/KG_data/FB15k-237-betae/train.txt", "r") as f:
#     train_graph = f.readlines()



from sentence_transformers import SentenceTransformer, util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b').to(device)


def fuse_mean(scores):
    ls = scores.unbind(dim = 0)
    res = torch.zeros(1,scores.shape[1])
    n = scores.shape[0]
    for l in ls:
        res += l / n 
    return res

def fuse_rrf(scores, k=10):
    ls = scores.unbind(dim=0)
    dict_scores = {}
    for l in ls:
        _, indices = torch.sort(l, descending=True)
        for i in range(scores.shape[1]):
            if indices[i].item() not in dict_scores:
                dict_scores[indices[i].item()] = []
            dict_scores[indices[i].item()].append(i + 1)

    for key in dict_scores:
        dict_scores[key] = sum([1/(k + rank) for rank in dict_scores[key]])
    scores = torch.tensor([list(dict_scores.values())])
    return scores



class ExpandSubgraph:
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
    rel2id = {v:k for k,v in id2rel.items()}  

    adj = None
    rel_embs = None
    def __init__(self, n_ent: int, n_rel: int, homoEdges: list, edge_index: list, args=None, depth=5, k=10, fuse_func=fuse_mean):
        self.model = model
        self.util = util
        self.k = k
        # print("type(edge_index):", type(edge_index))
        self.edge_index = edge_index
        if ExpandSubgraph.adj is None:
            ExpandSubgraph.adj = self.build_adjacency_list()
        if ExpandSubgraph.rel_embs is None:
            ExpandSubgraph.rel_embs = model.encode(list(ExpandSubgraph.id2rel.values()))
        self.depth = depth
        self.fuse_func = fuse_func
        self.subgraph_key = None
        self.answers_id = None
        self._name_cache = {}
        self.visited = set()

    def assign_query(self, query):
        self.queries = query['transformed_query']
        self.queries_emb = self.model.encode(self.queries)
        self.raw_query = query['raw_query']
        self.query_type = query['query_type']
        nums = extract_numbers(query['raw_query'])
        notations = extract_notations(query['query_type'])
        # print(nums, notations)
        
        self.start_entities = []
        for i in range(len(nums)):
            if notations[i] == 'e':
                self.start_entities.append(nums[i])

    def build_adjacency_list(self):
        """
        Currently separating direct and inverse edges.
        Optimized to use defaultdict for better performance.
        """
        print("Building adjacency list...")
        from collections import defaultdict
        adj = defaultdict(list)

        for i, edges in enumerate(self.edge_index):
            head, rel, tail = edges
            if head >= 14505 or tail >= 14505:
                print("Error: head or tail entity ID exceeds limit."
                      f" head: {head}, tail: {tail}")
                continue
            adj[head].append((rel, tail))
        
        # Convert back to regular dict if needed
        # print(list(adj.items())[0])
        # exit()
        return adj

    def id2name(self, idx):
        """Cached version to avoid repeated dictionary lookups."""
        idx = idx.item()
        if idx not in self._name_cache:
            self._name_cache[idx] = ExpandSubgraph.ent2name[ExpandSubgraph.id2ent[idx]]
        return self._name_cache[idx]

    def compare_rel_query_and_return_topk(self, triplets: np.array, queries, fuse_func=fuse_mean, ):
        # if type(triplets) is not list:
        #     triplets = [triplets]
        # if type(queries) is not list:
        #     queries = [queries] 
        
        # Get unique relations and create mapping
        unique_rel_ids = np.unique(triplets[:,1])
        rels_emb = ExpandSubgraph.rel_embs[unique_rel_ids]
        scores = self.util.dot_score(self.queries_emb, rels_emb)
        top_k_indices = self.return_top_k(scores, fuse_func=fuse_func)
        selected_rels = unique_rel_ids[top_k_indices]

        rel_order = {unique_rel_ids[idx]: rank for rank, idx in enumerate(top_k_indices)}
        filter_idx = (triplets[:, 1] == selected_rels[:, None]).any(axis=0)
        filtered_triplets = triplets[filter_idx]
        
        # Sort by the ranking from top-k
        sorted_keys = np.array([rel_order.get(item[1], float('inf')) for item in filtered_triplets])

        sorted_triplets = filtered_triplets[np.argsort(sorted_keys)]
        # print("sorted_triplets:", sorted_triplets)
        return sorted_triplets[:self.k]


    def return_top_k(self, scores, fuse_func=fuse_mean):
        scores = fuse_func(scores)
        k = min(self.k, scores.shape[1])
        scores, sorted_indices = torch.topk(scores, k, largest=True, dim=-1)
        # print("scores:", scores, "sorted_indices:", sorted_indices)
        return sorted_indices[0].tolist()

    def reset(self):
        self.subgraph_key = None
        self.visited = set()
        self._name_cache = {}

    def sampleSubgraph(self):
        assert (self.queries is not None), "Please assign a query first using assign_query()"
        # print("Sampling subgraph...")
        adjacency_list = ExpandSubgraph.adj
        start_entities = set(self.start_entities.copy())
        self.reset()
        
        for depth in range(self.depth):
            triplets = []
            for head_id in start_entities:
                if head_id not in adjacency_list.keys():
                    # print(f"No outgoing edges for entity {head}.")
                    continue
                edges = [(head_id, rel_id, tail_id) for rel_id, tail_id in adjacency_list[head_id]]
                triplets.extend(edges)

            triplets = [triplet for triplet in triplets if (triplet[0], triplet[1]) not in self.visited and self.id2name(triplet[2]) != "UnName_Entity"]
            #  print(len(triplets), "triplets found.")
            if triplets == []:
                continue
            triplets = self.compare_rel_query_and_return_topk(
                np.array(triplets), self.queries, fuse_func=self.fuse_func)
            if len(triplets) == 0:
                break
            if len(triplets) > self.k:
                sampling_idx = np.random.choice(np.arange(len(triplets)), size=self.k, replace=False)
                triplets = triplets[sampling_idx]

            for triplet in triplets:
                self.visited.add((triplet[0], triplet[1]))
                self.visited.add((triplet[2], triplet[1] + -1**(triplet[1] % 2 + 2)))  # Inverse relation
            
            if type(self.subgraph_key) is type(None):
                self.subgraph_key = triplets
                # print("Initial subgraph_key set.")
            else:
                
                self.subgraph_key = np.concatenate([self.subgraph_key, triplets], axis=0)

            ##### CHECK #####
            ents = np.unique(triplets[:, [0,2]].flatten())
            if (ents>=14505).any():
                print("Error: sampled entity ID exceeds limit.")
                print("Sampled triplets:", ents[ents>=14505])
                exit()
            #############
            start_entities = {triplet[2] for triplet in triplets}

        topk_nodes = np.unique(self.subgraph_key[:, [0,2]].flatten())
        # topk_nodes = list(topk_nodes)
        node_index = torch.zeros(len(ExpandSubgraph.id2ent), dtype=torch.long)
        node_index[topk_nodes] = torch.arange(len(topk_nodes))

        return topk_nodes, node_index, self.subgraph_key
        

    def evaluate_subgraph(self, type_eval="train"):
        if not self.subgraph_key:
            self.sampleSubgraph()

        subgraph_key = self.subgraph_key
        print("subgraph has", len(subgraph_key), "triplets.")

        entities_id = set()
        rels = set()
        for h, r, t in subgraph_key:
            entities_id.add(h)
            entities_id.add(t)
            rels.add(r)
        print("subgraph has", len(entities_id), "unique entities and", len(rels), "relation types.")
        entity_score = self.evaluate_ans_coverage(entities_id, type_eval=type_eval)
        relation_score = self.evaluate_rel_coverage(rels, type_eval=type_eval)
        return entity_score, relation_score

    def evaluate_ans_coverage(self, entities_id, type_eval="train"):
        answers_id = answer_train[self.raw_query].union(answer_valid_easy[self.raw_query])
        if type_eval == "valid" or type_eval == "test":
            answers_id = answers_id.union(answer_valid_hard[self.raw_query])
        if type_eval == "test": # chưa đụng tới
            answers_id = answers_id.union(answer_test_hard[self.raw_query])

        entity_score = len(answers_id.intersection(entities_id)) / len(answers_id)

        return entity_score

    def evaluate_rel_coverage(self, rels, type_eval="train"):
        # print(self.raw_query)
        notations = extract_notations(self.query_type)
        rels_ans = extract_numbers(self.raw_query)
        rels_ans = set([rel for i, rel in enumerate(rels_ans) if notations[i] == 'r'])

        relation_score = len(rels_ans.intersection(rels)) / len(rels_ans)
        return  relation_score
    
    
    def getOneSubgraph(self):
        assert (self.queries is not None), "Please assign a query first using assign_query()"
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
            ent_delta = sum(ent_delta_values)

            # Adding ent_delta to make node indices unique in the batch
            sampled_edges[:,0] = node_index[sampled_edges[:,0]] + ent_delta
            sampled_edges[:,2] = node_index[sampled_edges[:,2]] + ent_delta
            batch_sampled_edges.append(torch.from_numpy(sampled_edges))
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