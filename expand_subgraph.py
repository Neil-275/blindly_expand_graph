import openai 
import pickle as pkl
import random
from dotenv import load_dotenv
import prompt_list
import torch
from gen_query import extract_numbers, extract_strings, extract_notations
load_dotenv()


with open("/knowledge_graph/KG_data/FB15k-237-betae/train-queries.pkl", "rb") as f:
    queries = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/train-answers.pkl", "rb") as f:
    answer_train = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/valid-easy-answers.pkl", "rb") as f:
    answer_valid = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/test-easy-answers.pkl", "rb") as f:
    answer_test = pkl.load(f)

with open("/knowledge_graph/KG_data/FB15k-237-betae/train.txt", "r") as f:
    train_graph = f.readlines()

with open("/knowledge_graph/KG_data/FB15k-237-betae/id2ent.pkl", "rb") as f:
    id2ent = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/id2rel.pkl", "rb") as f:
    id2rel = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/ent2id.pkl", "rb") as f:
    ent2id = pkl.load(f)
with open("/knowledge_graph/KG_data/FB15k-237-betae/FB15k_mid2name.txt", "r") as f:
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
    def __init__(self, model, util, query, fuse_func=fuse_mean, depth=5, k=10):
        self.model = model
        self.util = util
        self.queries = query['transformed_query']
        self.raw_query = query['raw_query']
        nums = extract_numbers(query['raw_query'])
        notations = extract_notations(query['query_type'])
        # print(nums, notations)
        
        self.start_entities = []
        for i in range(len(nums)):
            if notations[i] == 'e':
                self.start_entities.append(nums[i])
        self.k = k
        self.adjacency_list = self.build_adjacency_list()
        self.depth = depth
        self.fuse_func = fuse_func
        self.subgraph_value = None
        self.subgraph_key = None
        self.answers_id = None

    def build_adjacency_list(self):
        adj = {}
        for edges in train_graph:
            head, rel, tail = edges.strip().split('\t')
            head, rel, tail = int(head), int(rel), int(tail)
            if head not in adj:
                adj[head] = []
            adj[head].append((rel, tail))
        return adj

    def id2name(self, idx):
        return ent2name[id2ent[idx]]

    def compare_rel_query_and_return_topk(self, triplets, queries, fuse_func=fuse_mean, k=5):
        if type(triplets) is not list:
            triplets = [triplets]
        if type(queries) is not list:
            queries = [queries] 
        # print(triplets)
        rels = [id2rel[rel_id] for _, rel_id, _ in triplets]

        rels_emb = self.model.encode(rels)
        queries_emb = self.model.encode(queries)
        scores = self.util.dot_score(queries_emb, rels_emb)
        top_k = self.return_top_k(scores, k=k, fuse_func=fuse_func)
        triplets = [triplets[idx] for idx in top_k]
        return triplets


    def return_top_k(self, scores, k=5, fuse_func=fuse_mean):
        scores = fuse_func(scores)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        return [idx for idx in sorted_indices.tolist()[0][:k]]

    def expand_subgraph(self):
        start_entities = self.start_entities.copy()
        self.subgraph_key = []

        for depth in range(self.depth):
            triplets = []
            print("depth:", depth)
            print(len(start_entities), "entities to expand.")
            for head_id in start_entities:
                head = self.id2name(head_id)
                if head == "UnName_Entity":
                    continue
                if head_id not in self.adjacency_list.keys():
                    print(f"No outgoing edges for entity {head}.")
                    continue
                edges = [(head_id, rel_id, tail_id) for rel_id, tail_id in self.adjacency_list[head_id]]
                triplets.extend(edges)

            triplets = [triplet for  triplet in triplets if self.id2name(triplet[2]) != "UnName_Entity"]
            print(len(triplets), "triplets found.")
            triplets = self.compare_rel_query_and_return_topk(triplets, self.queries, fuse_func=self.fuse_func, k=self.k)
            print(f"Selected {len(triplets)} triplets to expand.")
            if len(triplets) == 0:
                print("No more edges to expand.")
                break
            if len(triplets) > self.k:
                triplets = random.sample(triplets, self.k)
            self.subgraph_key.extend(triplets)
            start_entities = list(set([tail_id for _,_,tail_id in triplets if self.id2name(tail_id) != "UnName_Entity"]))
            # entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
            # print(f"Expanded to {len(start_entities)} entities.")

    def evaluate_subgraph(self, type_eval="train"):
        # pass
        if self.subgraph_key is None:
            self.expand_subgraph()
        print("subgraph has", len(self.subgraph_key), "triplets.")
        # print(self.subgraph_key[:3])
        self.answers_id = answer_train[self.raw_query]
        if type_eval == "valid" or type_eval == "test":
            self.answers_id = self.answers_id.union(answer_valid[self.raw_query])
        if type_eval == "test":
            self.answers_id = self.answers_id.union(answer_test[self.raw_query])
        
        self.answers_id = list(self.answers_id)

        entities_id = set()
        rels = set()
        for h,r,t in self.subgraph_key:
            entities_id.add(h)
            entities_id.add(t)
            rels.add(r)
        # return entities

        entity_score = 0
        for answer_id in self.answers_id:
            # print(answer)
            if answer_id  in entities_id:
                # print(f"Found answer entity {ent2name[answer]} in the subgraph.")
                entity_score += 1
        entity_score /= len(self.answers_id)
        relation_score = 0
        # for rel in rels:
        #     print(rel)
        #     rel = rel2id[rel]

            # if rel in rels:
            #     # print(f"Found answer relation {id2rel[rel]} in the subgraph.")
            #     relation_score += 1
        relation_score /= len(self.queries)
        return entity_score, relation_score
