import pickle as pkl
import random
from dotenv import load_dotenv
import prompt_list
import torch
from gen_query import extract_numbers, extract_strings, extract_notations
from typing import Literal
load_dotenv()


with open("knowledge_graph/KG_data/FB15k-237-betae/train-queries.pkl", "rb") as f:
    queries = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/train-answers.pkl", "rb") as f:
    answer_train = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/valid-easy-answers.pkl", "rb") as f:
    answer_valid_easy = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/valid-hard-answers.pkl", "rb") as f:
    answer_valid_hard = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/test-easy-answers.pkl", "rb") as f:
    answer_test_easy = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/test-hard-answers.pkl", "rb") as f:
    answer_test_hard = pkl.load(f)

with open("knowledge_graph/KG_data/FB15k-237-betae/train.txt", "r") as f:
    train_graph = f.readlines()

with open("knowledge_graph/KG_data/FB15k-237-betae/id2ent.pkl", "rb") as f:
    id2ent = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/id2rel.pkl", "rb") as f:
    id2rel = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/ent2id.pkl", "rb") as f:
    ent2id = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/FB15k_mid2name.txt", "r") as f:
    ent2name = {}
    for line in f:
        mid, name = line.strip().split("\t")
        ent2name[mid] = name

name2ent = {v:k for k,v in ent2name.items()}  
rel2id = {v:k for k,v in id2rel.items()}  

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

    dir_adj, inv_adj = None, None
    def __init__(self, model, util, query, fuse_func=fuse_mean, depth=5, k=10):
        self.model = model
        self.util = util
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
        self.k = k
        if ExpandSubgraph.dir_adj is None or ExpandSubgraph.inv_adj is None:
            ExpandSubgraph.dir_adj, ExpandSubgraph.inv_adj = self.build_adjacency_list()
        self.depth = depth
        self.fuse_func = fuse_func
        self.subgraph_value = None
        self.subgraph_key = {'direct': [], 'inverse': []}
        self.answers_id = None
        self._name_cache = {}

    def build_adjacency_list(self):
        """
        Currently separating direct and inverse edges.
        Optimized to use defaultdict for better performance.
        """
        print("Building adjacency list...")
        from collections import defaultdict
        
        dir_adj = defaultdict(list)
        inv_adj = defaultdict(list)
        
        for i, edges in enumerate(train_graph):
            head, rel, tail = map(int, edges.strip().split('\t'))
            if i % 2 == 0:
                dir_adj[head].append((rel, tail))
            else:
                inv_adj[head].append((rel, tail))
        
        # Convert back to regular dict if needed
        return dir_adj, inv_adj

    def id2name(self, idx):
        """Cached version to avoid repeated dictionary lookups."""
        if idx not in self._name_cache:
            self._name_cache[idx] = ent2name[id2ent[idx]]
        return self._name_cache[idx]

    def compare_rel_query_and_return_topk(self, triplets, queries, fuse_func=fuse_mean, ):
        if type(triplets) is not list:
            triplets = [triplets]
        if type(queries) is not list:
            queries = [queries] 
        
        # Get unique relations and create mapping
        unique_rel_ids = list(set(triplet[1] for triplet in triplets))
        rel_names = [id2rel[rel_id] for rel_id in unique_rel_ids]

        rels_emb = self.model.encode(rel_names)
        scores = self.util.dot_score(self.queries_emb, rels_emb)
        top_k_indices = self.return_top_k(scores, fuse_func=fuse_func)
        selected_rels = [rel_names[idx] for idx in top_k_indices]
        rel_order = {rel_names[idx]: rank for rank, idx in enumerate(top_k_indices)}
        # print("rel_order:", rel_order)
        # Filter and sort triplets
        filtered_triplets = [triplet for triplet in triplets 
                            if id2rel[triplet[1]] in selected_rels]
        
        # Sort by the ranking from top-k
        sorted_triplets = sorted(filtered_triplets, 
                            key=lambda x: rel_order.get(id2rel[x[1]], float('inf')))
        # print("sorted_triplets:", sorted_triplets)
        return sorted_triplets[:self.k]


    def return_top_k(self, scores, fuse_func=fuse_mean):
        scores = fuse_func(scores)
        k = min(self.k, scores.shape[1])
        scores, sorted_indices = torch.topk(scores, k, largest=True, dim=-1)
        # print("scores:", scores, "sorted_indices:", sorted_indices)
        return sorted_indices[0].tolist()

    def expand_subgraph(self, adj_type: Literal["direct", "inverse"] = "direct"):
        # print("Expanding", adj_type, "subgraph...")
        if adj_type == "direct":
            adjacency_list = self.dir_adj
        else:
            adjacency_list = self.inv_adj
        
        start_entities = set(self.start_entities.copy())
        self.subgraph_key[adj_type] = []

        for depth in range(self.depth):
            triplets = []
            # print("depth:", depth, end="\t")
            # print(len(start_entities), "entities to expand.")
            for head_id in start_entities:
                # head = self.id2name(head_id)
                # if head == "UnName_Entity":
                #     continue
                if head_id not in adjacency_list.keys():
                    # print(f"No outgoing edges for entity {head}.")
                    continue
                edges = [(head_id, rel_id, tail_id) for rel_id, tail_id in adjacency_list[head_id]]
                triplets.extend(edges)

            triplets = [triplet for  triplet in triplets if self.id2name(triplet[2]) != "UnName_Entity"]
            # print(len(triplets), "triplets found.")
            if triplets == []:
                # print("No triplets to expand.")
                continue
            triplets = self.compare_rel_query_and_return_topk(
                triplets, self.queries, fuse_func=self.fuse_func)
            # print(f"Selected {len(triplets)} triplets to expand.")
            if len(triplets) == 0:
                # print("No more edges to expand.")
                break
            if len(triplets) > self.k:
                triplets = random.sample(triplets, self.k)
            self.subgraph_key[adj_type].extend(triplets)
            start_entities = {triplet[2] for triplet in triplets}

    def evaluate_subgraph(self, type_eval="train"):
        if 'direct' not in self.subgraph_key or not self.subgraph_key['direct']:
            self.expand_subgraph("direct")
        if 'inverse' not in self.subgraph_key or not self.subgraph_key['inverse']:
            self.expand_subgraph("inverse")
        subgraph_key = self.subgraph_key['direct'] + self.subgraph_key['inverse']
        # print("subgraph has", len(subgraph_key), "triplets.")
        # print(self.subgraph_key[:3])
        entities_id = set()
        rels = set()
        for h,r,t in subgraph_key:
            entities_id.add(h)
            entities_id.add(t)
            rels.add(r)
        # print("subgraph has", len(entities_id), "unique entities and", len(rels), "relation types.")
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