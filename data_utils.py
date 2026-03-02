import pickle as pkl

with open("knowledge_graph/KG_data/FB15k-237-betae/id2ent.pkl", "rb") as f:
    id2ent = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/id2rel.pkl", "rb") as f:
    id2rel = pkl.load(f)
with open("knowledge_graph/KG_data/FB15k-237-betae/ent2id.pkl", "rb") as f:
    ent2id = pkl.load(f)

with open("knowledge_graph/queries/train_all_id.pkl", "rb") as f:
    queries = pkl.load(f)

with open("knowledge_graph/KG_data/FB15k-237-betae/FB15k_mid2name.txt", "r", encoding='utf-8') as f:
    ent2name = {}
    for line in f:
        mid, name = line.strip().split("\t", 1)  # Use maxsplit=1 in case name has tabs
        ent2name[mid] = name
        

name2ent = {v:k for k,v in ent2name.items()}  
rel2id = {v:k for k,v in id2rel.items()} 