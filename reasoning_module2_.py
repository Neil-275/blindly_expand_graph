import random
import re
import numpy as np
import torch
from dotenv import load_dotenv

from utils import (
    run_llm, extract_numbers, extract_notations, extract_answer,
    get_subgraph_relation_embeddings,
    get_last_token_embedding
    )
from prompt_list2 import decision_prompt, entities_pruning_prompt, GoG_answer_prompt
from extract_llm_answers import extract_decision_json, extract_pruned_entities
from utils_for_reasoning import map_to_most_similar, map_to_most_similar_list
from data_utils import id2ent, ent2name, id2rel, rel2id
import sys
load_dotenv()


class ReasoningModule2:
    def __init__(self, sampler, gnn_model, projector, model_args=None, k=4, max_hops=4):
        self.sampler = sampler
        self.gnn_model = gnn_model
        self.projector = projector
        self.model_args = model_args
        self.k = k
        self.max_hops = max_hops
        self.result = None
        self.query = None

    def assign_query(self, query):
        self.query = query
        self.sampler.assign_query(query)
        self.subgraph_data = self.sampler.sampleSubgraph()
        ## Convert to local indexing for GNN processing
        # global_rel = subgraph_data[2][:,1] 
        # _, reverse_out = torch.unique(global_rel, return_inverse=True)
        # subgraph_data[2][:,1] = reverse_out

        # self.subgraph_data = subgraph_data
        # self.subgraph_nodes, _, self.subgraph_edges = subgraph_data
        # self.rel_emb = self.construct_rel_emb(self.subgraph_edges)
        # self.gnn_model.assign_rel_emb(self.rel_emb)
        # self.adj_list = self.build_adjacency_list()
        
    def assign_dict(self, id2ent, id2rel, rel_desc_dict):
        self.id2ent, self.id2rel = id2ent, id2rel
        self.ent2id = {v:k for k,v in id2ent.items()}
        self.rel2id = {v:k for k,v in id2rel.items()}
        self.rel_desc_dict = rel_desc_dict
        ## Dummy: Delete later
        global_rel = self.subgraph_data[2][:,1] 
        _, reverse_out = torch.unique(global_rel, return_inverse=True)
        self.subgraph_data[2][:,1] = reverse_out

        self.subgraph_nodes, _, self.subgraph_edges = self.subgraph_data
        self.rel_emb = self.construct_rel_emb(self.subgraph_edges)
        self.gnn_model.assign_rel_emb(self.rel_emb)
        self.adj_list = self.build_adjacency_list()

    def _get__start_entities_from_query(self, query):
        numbers = extract_numbers(query['raw_query'])
        notations = extract_notations(query['query_type'])
        return [numbers[i] for i in range(len(numbers)) if notations[i] == 'e']
    
    def build_adjacency_list(self):
        """
        Currently separating direct and inverse edges.
        Optimized to use defaultdict for better performance.
        """
        # print("Building adjacency list...")
        from collections import defaultdict
        adj = defaultdict(list)

        for i, edges in enumerate(self.subgraph_edges.cpu().numpy()):
            head, rel, tail = edges
            adj[head].append((rel.item(), tail.item()))
        
        return adj

    def convert_to_local_indexing(self):
        """
        Convert subgraph data from global indexing to local indexing.
        
        Returns:
            tuple: (local_nodes, local_node_mapping, local_edges)
                - local_nodes: tensor of local indices (0 to len(topk_nodes)-1)
                - local_node_mapping: dict mapping global node id to local index
                - local_edges: tensor of edges with local node indices and global relation indices
        """
        topk_nodes, node_index, subgraph_edges = self.subgraph_data
        
        # Create mapping from global node ID to local index
        global_to_local = {}
        for local_idx, global_node_id in enumerate(topk_nodes.tolist()):
            global_to_local[global_node_id] = local_idx
        
        # Create local node indices (0, 1, 2, ..., len(topk_nodes)-1)
        local_nodes = torch.arange(len(topk_nodes))
        
        # Convert edges to local indexing
        local_edges_list = []
        for edge in subgraph_edges:
            head_global, relation_id, tail_global = edge.tolist()
            
            # Only keep edges where both nodes are in our subgraph
            if head_global in global_to_local and tail_global in global_to_local:
                head_local = global_to_local[head_global]
                tail_local = global_to_local[tail_global]
                # Keep relation indices as global (since we might need them for embeddings)
                local_edges_list.append([head_local, relation_id, tail_local])
        
        local_edges = torch.tensor(local_edges_list) if local_edges_list else torch.empty((0, 3), dtype=torch.long)
        
        return local_nodes, global_to_local, local_edges

    def construct_rel_emb(self, subgraph_edges):
        rel_emb = get_subgraph_relation_embeddings(subgraph_edges)
        sorted_rels = sorted(rel_emb.keys())
        rel_emb_tensor = torch.stack([rel_emb[r] for r in sorted_rels])
        return rel_emb_tensor
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_1hop_relations(self, entity_id):
        """Return list of NL relation names reachable in one hop from entity."""
        rel_unique = list(set(r_id for r_id, _ in self.adj_list.get(entity_id, [])))
        return [f"{r_id}: {self.rel_desc_dict[r_id]}" for r_id in rel_unique]

    def _trace_to_triplet_strings(self, trace):
        """Convert a path trace to human-readable triplet strings for prompts.

        Each trace entry is (head_entity_id, via_relation_nl, tail_entity_id).
        """
        return [
            f"{self.id2ent[h]}, {r}, {self.id2ent[t]}"
            for h, r, t in trace
        ]

    def _trace_to_history_string(self, trace):
        """Convert trace to show only the last hop that led to the current entity for the pathfinder prompt."""
        if not trace:
            return "No active reasoning path yet."
        # Get only the last hop: [Last-iter entity] --> [Relation] --> [Current entity]
        h, r, t = trace[-1]

        return f"{self.id2ent[h]} --> {r} --> {self.id2ent[t]}"

    # ------------------------------------------------------------------
    # Step 1 – JUDGE
    # ------------------------------------------------------------------

    def call_llm_judge(self, question, knowledge_triplets):
        """Check whether the current trace already contains the answer.

        Uses GoG_answer_prompt.
        Returns dict: {'is_goal_reached': bool, 'final_answer': str | None}
        """
        def if_true(s):
            return s.lower().strip().replace(" ", "") == "yes"

        if not knowledge_triplets:
            return {"is_goal_reached": False, "final_answer": None}

        q = question[0] if isinstance(question, list) else question
        # print("length of knowledge_triplets:", len(knowledge_triplets))
        # print(knowledge_triplets[0])
        # traces = [kt['trace'] for kt in knowledge_triplets]
        traces = knowledge_triplets['trace']
        # print(traces)
        triplet_strs = self._trace_to_triplet_strings(traces)
        prompt = (
            GoG_answer_prompt
            + q 
            + "\nKnowledge Triplets: \n" + "\n".join(triplet_strs)
            + '\nA: '
        )
        
        response = run_llm(prompt, engine=self.model_args)
        print("------ call_llm_judge:")
        # print("Prompt:", prompt)
        # print("Response:", response)
        if if_true(extract_answer(response)):
            return {"is_goal_reached": True, "final_answer": response}
        return {"is_goal_reached": False, "final_answer": None}

    # ------------------------------------------------------------------
    # Step 2 – PATHFINDER
    # ------------------------------------------------------------------

    def call_llm_pathfinder(self, question, entity_id, trace, k= 3):
        """Ask LLM which relations to follow from current entity.

        Uses decision_prompt with the full reasoning history from the trace.
        Returns parsed decision dict with 'contributory_relations'.
        """
        q = question[0] if isinstance(question, list) else question
        local_relations = self.get_1hop_relations(entity_id)
        history_str = self._trace_to_history_string(trace)

        prompt = decision_prompt
        prompt = prompt.replace("{{question}}", q)
        prompt = prompt.replace("{{current_entity}}", ent2name[id2ent[entity_id]])
        prompt = prompt.replace("{{reasoning_history}}", history_str)
        prompt = prompt.replace("{{context}}", "\n".join(local_relations))
        prompt = prompt.replace("{{k}}", str(k))
        prompt += '\nA: '

        response = run_llm(prompt, engine=self.model_args)
        
        # if history_str != "No active reasoning path yet.":
        print("------ call_llm_pathfinder:")
        #     print("Prompt:", prompt)
        #     print("Response:", response)
        return extract_decision_json(response)

    # ------------------------------------------------------------------
    # Step 3 – GNN (Semantic Link Prediction)
    # ------------------------------------------------------------------

    def prepareInput_for_GNN(self, entity_id):

        def prepareData(batch_data):
            subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
            subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
            subs = subs.cuda().flatten()
            rels = rels.cuda().flatten()
            objs = objs.cuda()
            return subs, rels, objs, subgraph_data

        tmp_node = torch.tensor(entity_id).unsqueeze(0)
        tmp_relation = torch.tensor([0]).unsqueeze(0) # dummy tensor for relation (not used in GNN input, but required by prepareData)

        topk_nodes, node_index, subgraph_edges = self.subgraph_data

        batch_subgraph = self.sampler.getBatchSubgraph([[tmp_node, topk_nodes.clone(), node_index.clone(), subgraph_edges.clone()]])
        
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_subgraph
        dummy_objs = torch.tensor([0])  # Dummy tensor for objs
        subs, rels, _, subgraph_data = prepareData([tmp_node, tmp_relation, dummy_objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges])
        return subs, rels, subgraph_data

    def gnn_link_prediction(self, entity_id, rel_emb: torch.tensor, existing_rel_id: int=None):
        """Score all subgraph nodes given (entity, NL_relation).

        Maps the NL relation description to the closest real KG relation,
        then runs the GNN. Works for both 'existing' and 'missing' relations.

        Returns dict {node_global_id: score}.
        """
        
        subs, _, subgraph_data = self.prepareInput_for_GNN(entity_id)
        rel_emb = rel_emb.cuda().unsqueeze(0)  # Shape: (1, embedding_dim)
        gnn_scores = self.gnn_model.forward_inference(subs,  rel_emb, subgraph_data, self.projector)

        # Boost scores for nodes directly connected via this relation in the KG
        max_score = gnn_scores.max().item()
        if existing_rel_id is not None:
            # print(existing_rel_id, 123)
            kg_edges = self.subgraph_edges.cpu().numpy()
            for _, tail in self.adj_list.get(entity_id, []):
                for triple in [
                    np.array([entity_id, existing_rel_id, tail]),
                    np.array([tail, existing_rel_id + 1, entity_id]),
                ]:
                    if np.any(np.all(triple == kg_edges, axis=1)):
                        idx = torch.argwhere(self.subgraph_nodes == tail).squeeze()
                        if idx.numel() == 1:
                            # print(f"Boosting node {tail} for relation {existing_rel_id}")
                            gnn_scores[idx.item()] = max_score + 10.0

        return {
            node_id: gnn_scores[i].item()
            for i, node_id in enumerate(self.subgraph_nodes.tolist())
        }

    # ------------------------------------------------------------------
    # Step 4 – PRUNER
    # ------------------------------------------------------------------

    def call_llm_filter(self, question, cur_ent, rel_nl, candidate_pool, k):
        """Use LLM to prune candidate_pool down to top-k entities.

        Uses entities_pruning_prompt.
        Returns list of up to k candidate dicts from candidate_pool.
        """
        if not candidate_pool:
            return []

        q = question[0] if isinstance(question, list) else question
        pool_str = "; ".join(
            f"{ent2name[id2ent[c['entity']]]}"
            for c in candidate_pool
        )
        prompt = entities_pruning_prompt
        prompt = prompt.replace("{{question}}", q)
        prompt = prompt.replace("{{current_entity}}", ent2name[id2ent[cur_ent]])
        prompt = prompt.replace("{{current_relation}}", rel_nl)
        prompt = prompt.replace("{{candidate_pool}}", pool_str)
        prompt = prompt.replace("{{k}}", str(k))
        prompt += '\nA: '

        response = run_llm(prompt, engine=self.model_args)

        print("------ call_llm_filter:")
        # print("Prompt:", prompt)
        # print("Response:", response)
        # sys.exit()
        selected_names = extract_pruned_entities(response)

        # Map returned names back to candidate dicts (best score per name)
        name_to_cand = {}
        for c in candidate_pool:
            name = ent2name[id2ent[c["entity"]]]
            if name not in name_to_cand or c["score"] > name_to_cand[name]["score"]:
                name_to_cand[name] = c

        result = []
        for name in selected_names:
            matched = map_to_most_similar(name, list(name_to_cand.keys()), threshold=0.3)
            if matched:
                cand = name_to_cand[matched]
                if cand not in result:
                    result.append(cand)

        # Fallback: top-k by GNN score if LLM returns nothing usable
        if not result:
            result = sorted(candidate_pool, key=lambda x: x["score"], reverse=True)[:k]

        return result[:k]

    # ------------------------------------------------------------------
    # Main reasoning loop
    # ------------------------------------------------------------------

    def reasoning(self):
        question = self.query['natural_query']
        k = self.k
        knowledge_triplets = []  # To keep track of the reasoning path
        active_paths = [
            {"current_entity": te, "trace": []}
            for te in self._get__start_entities_from_query(self.query)
        ]
        self.result = None

        for hop in range(self.max_hops):
            new_active_paths = []
            print("Hop", hop + 1, "with", len(active_paths), "active path(s)")
            # judge_decision = self.call_llm_judge(question, knowledge_triplets)
            # if judge_decision["is_goal_reached"]:
            #     print("Answer found:", judge_decision["final_answer"])
            #     self.result = judge_decision["final_answer"]
            #     return

            for i_path, path in enumerate(active_paths):
                print(f"path: {i_path}", path)
                entity_id = path["current_entity"]

                # 1. JUDGE: Does the current trace already answer the question?
                judge_decision = self.call_llm_judge(question, path)
                if judge_decision["is_goal_reached"]:
                    print("Answer found:", judge_decision["final_answer"])
                    self.result = judge_decision["final_answer"]
                    return

                # 2. PATHFINDER: Which relations should we follow from here?
                search_directions = self.call_llm_pathfinder(question, entity_id, path["trace"])
                existing = search_directions.get("contributory_relations", {}).get("existing", [])
                missing  = search_directions.get("contributory_relations", {}).get("missing",  [])
                existing_rels = [(r_id, self.rel_desc_dict[r_id]) for r_id in existing]
                missing_rels = [(None, r) for r in missing ]
                all_target_relations = existing_rels + missing_rels 

                # 3. GNN: Score candidate nodes for every target relation
                selected_candidates = []
                for rel_id, rel_nl in all_target_relations:
                    candidate_per_rel_pool = []
                    rel_emb = get_last_token_embedding(rel_nl)
                    # print(rel_id, rel_nl, 123)
                    # print("rel_nl:", rel_nl)
                    gnn_scores = self.gnn_link_prediction(entity_id, rel_emb, rel_id)
                    top_nodes = sorted(gnn_scores.items(), key=lambda x: x[1], reverse=True)[: 2 * k]
                    for node_id, score in top_nodes:
                        candidate_per_rel_pool.append({
                            "entity": node_id,
                            "via_relation": rel_nl,
                            "score": score,
                        })

                    if not candidate_per_rel_pool:
                        continue
                    # print(candidate_pool)
                    # 4. PRUNER: Filter candidates to extend this path
                    selected_candidates_per_rel = self.call_llm_filter(
                        question, entity_id, rel_nl, candidate_per_rel_pool, k
                    )
                    selected_candidates.extend(selected_candidates_per_rel)
                if len(selected_candidates) > self.k: # Select randomly if too many candidates from multiple relations
                    # candidate_pool = candidate_pool[:self.k]
                    selected_candidates = random.sample(selected_candidates, self.k)  
            
                for cand in selected_candidates:
                    new_active_paths.append({
                        "current_entity": cand["entity"],
                        "trace": path["trace"] + [
                            (entity_id, cand["via_relation"], cand["entity"])
                        ],
                    })
            # print(len(new_active_paths), "new paths generated at hop", hop + 1)
            if not new_active_paths:
                print("No new paths to explore.")
                break

            # Limit total active paths to avoid combinatorial explosion
            active_paths = new_active_paths[: k * 2]
            knowledge_triplets.extend(active_paths)
            # print(f"Hop {hop + 1}: {len(active_paths)} active path(s)")

        print("Max hops reached without a conclusive answer.")
        self.result = []
