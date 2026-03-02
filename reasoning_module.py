import openai
import os
from dotenv import load_dotenv
import pickle as pkl
from utils import extract_numbers, extract_notations
from prompt_list2 import *
import re
from utils import run_llm, extract_notations, extract_numbers, extract_answer
import numpy as np
import torch

from utils_for_reasoning import (
    map_to_most_similar,
    map_to_most_similar_list,
    construct_relation_prune_prompt,
    construct_decision_prompt,
    extract_decision,
    clean_relations,
)
from extract_llm_answers import extract_decision_json

from data_utils import id2ent, ent2name, id2rel, rel2id

load_dotenv()

client = openai.OpenAI()



class ReasoningModule:
    def __init__(self,  sampler, subgraph_data,  gnn_model,  model_args: str = None, k = 4, m_candidates = 3, max_depth = 4):
        self.sampler = sampler
        self.subgraph_data = subgraph_data
        self.subgraph_nodes, _, self.subgraph_edges = self.subgraph_data
        self.model_args = model_args
        self.gnn_model = gnn_model
        self.rel_list = self.get_relations_from_subgraph()
        self.k = k
        self.m_candidates = m_candidates
        self.max_depth = max_depth
        self.result = None
        self.cands = None
    
    def assign_query(self, query):
        self.query = query
        self.entities = self.get_entities_from_query(query)

    def assign_subgraph(self, subgraph_data):
        self.subgraph_data = subgraph_data
        self.subgraph_nodes, _, self.subgraph_edges = self.subgraph_data
        self.adj_list = self.sampler.adj_list
        self.rel_list = self.get_relations_from_subgraph()
         

    def get_relations_from_subgraph(self):
        """
        Currently just get all relations in the subgraph.
        Only use the direct relations ("+/") because this follows the real pattern in the knowledge graph.

        Future work: 
        + Maybe use LLM to predict the relevant relations based on the query and entities.
        """
        relations = torch.unique(self.subgraph_edges[:,1]).tolist()
        res = []
        for i in range(len(relations)):
            if "+/" in id2rel[relations[i]]:
                res.append(relations[i])
        return res

    def get_entities_from_query(self, query):
        numbers = extract_numbers(query['raw_query'])
        notations = extract_notations(query['query_type'])
        entities = []
        for i in range(len(numbers)):
            if notations[i] == 'e':
                # notations[i] = numbers[i]
                entities.append(numbers[i])
        return entities

    
    def merge_item_by_key(self, list_dict, key, ops='max'):
        """
        Merge two lists of relation dictionaries, keeping the one with higher score for each relation.
        Each dict should have  and 'score'.
        """
        # Create a dictionary to track relations and their best scores
        merged_dict = {}
        for item in list_dict:
            mkey = item[key]
            if item[key] not in merged_dict.keys():
                merged_dict[mkey] = item
            else:
                if ops == "max":
                    merged_dict[mkey]['score'] = max(merged_dict[mkey]['score'], item['score'])
                elif ops == "sum":
                    merged_dict[mkey]['score'] += item['score']
        # print(merged_dict)
        return list(merged_dict.values())
    

    
    def relation_search(self, entity_id):
        """
            ask LLM with the decision prompt to prune the relations for the given entity_id, return a list of relations to follow.
        """
        context_rel = [id2rel[r_id] for e_id, r_id in self.adj_list.get(entity_id, [])]
        prompt = construct_decision_prompt(self.query['natural_query'], ent2name[id2ent[entity_id]], context_rel)

        response = run_llm(prompt, engine=self.model_args)
        parsed = extract_decision_json(response)

        existing_nl_relations = parsed.get("contributory_relations", {}).get("existing", [])
        matched_relations = map_to_most_similar_list(existing_nl_relations, context_rel)

        seen = set()
        result = []
        for rel in matched_relations:
            if rel is None or rel in seen:
                continue
            seen.add(rel)
            result.append({
                "entity_id": entity_id,
                "relation": rel,
                "score": 1.0,
                "head": rel.startswith("+"),
            })

        if not result:
            default_score = 1.0 / len(context_rel) if context_rel else 0.0
            for rel in context_rel:
                result.append({
                    "entity_id": entity_id,
                    "relation": rel,
                    "score": default_score,
                    "head": rel.startswith("+"),
                })

        return result



    def check_answer_sufficiency(self, cluster_chain_of_entities: list, query, args):
        """
        Use LLM to check whether the current knowledge triplets are sufficient to answer the question.
        """
        def if_true(prompt):
            if prompt.lower().strip().replace(" ","")=="yes":
                return True
            return False
        prompt = GoG_answer_prompt + query['natural_query'] + '\n'
        cluster_chain_of_entities = [", ".join([ent2name[id2ent[e_id]], id2rel[r_id], ent2name[id2ent[c_id]]]) for e_id, r_id, c_id in cluster_chain_of_entities]
        chain_prompt = "\n".join(cluster_chain_of_entities)
        prompt += "\nKnowledge Triplets: " + chain_prompt + '\nA: '
        # print(f"#### reasoning_prompt: {prompt}")
        response = run_llm(prompt, engine=args)
        
        result = extract_answer(response)
        if if_true(result):
            return True, response
        else:
            return False, response



    def prepareInput_for_GNN(self, entity_id, relation):

        def prepareData(batch_data):
            subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
            subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
            subs = subs.cuda().flatten()
            rels = rels.cuda().flatten()
            objs = objs.cuda()
            return subs, rels, objs, subgraph_data

        tmp_node = torch.tensor(entity_id).unsqueeze(0)
        tmp_relation = torch.tensor(rel2id[relation]).unsqueeze(0)

        topk_nodes, node_index, subgraph_edges = self.subgraph_data

        batch_subgraph = self.sampler.getBatchSubgraph([[tmp_node, topk_nodes.clone(), node_index.clone(), subgraph_edges.clone()]])
        
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_subgraph
        dummy_objs = torch.tensor([0])  # Dummy tensor for objs
        subs, rels, _, subgraph_data = prepareData([tmp_node, tmp_relation, dummy_objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges])
        return subs, rels, subgraph_data

    

    def filter_cands(self, candidate_ids, query, entity_id, relation, scores, m_candidates, args):
        def clean_scores(string, entity_candidates):
            names = re.findall(r'\+ (.+?):', string)
            scores = re.findall(r'\d+\.\d+', string)
            scores = [float(number) for number in scores]
            if len(scores) == len(entity_candidates) and names == entity_candidates:
                return scores
            else:
                print("All entities are created equal.")
                return [1/len(entity_candidates)] * len(entity_candidates)
            
        def construct_entity_score_prompt(query, relation, entity_candidates, scores):
            
            return entities_pruning_prompt.format(query , relation) + "; ".join(entity_candidates) + '\nScore: '
        
        #########
   
        entity_candidates = [ent2name[id2ent[candidate_id]] for candidate_id in candidate_ids]
        prompt = construct_entity_score_prompt(query['natural_query'], relation, entity_candidates, scores)
        result = run_llm(prompt, engine=args)
        # print(res_entity[0])
        # print("A: ", entity_candidates)
        # print("B:", res_entity)

        clean_relations = clean_scores(result, entity_candidates)
        while clean_relations == -1:
            print("invalid entity pruning, retrying...")
            result = run_llm(prompt, engine=args)
            clean_relations = clean_scores(result, entity_candidates)    
        print("valid entity pruning")
        return [float(x) * scores[i] + 1e-10 for i,x in enumerate(clean_relations)], entity_candidates, candidate_ids

    def reasoning(self):
        # print("Start reasoning...")
        info = []
        self.cands = []
        beams = [{"entity_id": entity_id, "score": 1.0} for entity_id in self.entities]
        i_loop = 0
        visited = torch.ones(len(self.subgraph_nodes), dtype=torch.bool)
        while True:
            all_candidates = []
            for beam in beams:
                # print("Current beam:", ent2name[id2ent[beam["entity_id"]]])
                beam_candidates = []
                entity_id = beam["entity_id"]
                # print(entity_id)
                entity_score = beam["score"]
                if visited[torch.argwhere(self.subgraph_nodes == entity_id).squeeze().item()] == 0:
                    continue
                visited[torch.argwhere(self.subgraph_nodes == entity_id).squeeze().item()] = 0
                # print(beam)
                
                current_entity_relations_list = self.relation_search(entity_id) # {"entity_id": entity_id, "relation": relation, "score": score, "head": True/False}
                # print("current_entity_relations_list:", current_entity_relations_list)
                # current_entity_relations_list = [{'entity_id': 8552, 'relation': '+/education/educational_institution/students_graduates./education/education/student', 'score': 0.5, 'head': True}, {'entity_id': 8552, 'relation': '+/education/educational_institution/school_type', 'score': 0.3, 'head': True}, {'entity_id': 8552, 'relation': '+/education/educational_institution/colors', 'score': 0.1, 'head': True}, {'entity_id': 8552, 'relation': '+/organization/organization/headquarters./location/mailing_address/state_province_region', 'score': 0.1, 'head': True}]

                for tmp in current_entity_relations_list[::]: # loop through relations
                    try: ### make sure the relation is in the rel2id
                        if rel2id[tmp['relation']] is None:
                            continue
                    except:
                        print("Relation not found in rel2id:", tmp['relation'])
                        continue
                    print(f"_____________________\nProcessing relation: {tmp['relation']}")
                    rel_score = tmp['score']  # Get the relation score
                    subs, rels, subgraph_data = self.prepareInput_for_GNN(entity_id, tmp['relation'])
                    # print(123)
                    gnn_scores_all = self.gnn_model(subs, rels, subgraph_data, score_all=False) # compute scores for candidates under this relation
                    gnn_scores_all = gnn_scores_all.squeeze()
                    # print("len(gnn_scores_all):", len(gnn_scores_all))
                    # exit()
                    max_gnn_score = gnn_scores_all.max().item()
                    for rel, tail in self.adj_list.get(entity_id, []):
                        # Check cả 2 hướng, nếu có edge thì set score = 1.0
                        triple = np.array([entity_id, rel2id[tmp['relation']], tail])
                        if (np.any(np.all(triple == self.subgraph_edges.cpu().numpy(), axis=1))):
                            i = torch.argwhere(self.subgraph_nodes == tail).squeeze().item()
                            gnn_scores_all[i] = max_gnn_score + 10.0
                            # print(triple)
                        triple_rev = np.array([tail, rel2id[tmp['relation']]+ 1, entity_id])
                        if (np.any(np.all(triple_rev == self.subgraph_edges.cpu().numpy(), axis=1))):
                            i = torch.argwhere(self.subgraph_nodes == tail).squeeze().item()
                            gnn_scores_all[i] = max_gnn_score + 10.0
                    
                    gnn_scores_all = gnn_scores_all * visited.cuda()  # only consider unvisited nodes
                    # print(gnn_scores_all.shape)
                    # gnn_scores_all = torch.argwhere(gnn_scores_all > 0)  # filter out zero scores

                    _ , topk_candidate_ids = torch.topk(gnn_scores_all, self.m_candidates * 2, largest=True)
                    # gnn_scores_all[topk_candidate_ids] = torch.softmax(gnn_scores_all[topk_candidate_ids], dim=0) # normalize scores                    
                    topk_candidate_ids = topk_candidate_ids.cpu().tolist()
                    print("@@@@@@@@@@@@@@  \n Top candidates from GNN:" )

                    
                    candidates = []
                    for candidate_id in topk_candidate_ids:
                        gnn_score = gnn_scores_all[candidate_id].item() + 1e-10  # avoid zero scores
                        print(f"{ent2name[id2ent[self.subgraph_nodes[candidate_id].item()]]} | score: ", gnn_score)
                        candidates.append({"entity_id": entity_id, "relation": rel2id[tmp['relation']], "candidate_id": self.subgraph_nodes[candidate_id].item(), "score": gnn_score * rel_score * entity_score})
                    filtered_candidates = self.filter_cands([a['candidate_id'] for a in candidates], self.query, entity_id, tmp['relation'], [a['score'] for a in candidates], self.m_candidates, self.model_args,)
                    print("############\n Top candidates after LLM filtering:" )
                    filtered_candidates = sorted(zip(filtered_candidates[0], filtered_candidates[2]), key=lambda x: x[0], reverse=True)[:self.m_candidates]
                    for score, candidate_id in filtered_candidates:
                        print(f"{ent2name[id2ent[candidate_id]]} | score: ", score)
                        beam_candidates.append({"entity_id": entity_id, "relation": rel2id[tmp['relation']], "candidate_id": candidate_id, "score": score})
                    # beam_candidates.extend(filtered_candidates[2])
                # me
                # print("summarize")
                beam_candidates = self.merge_item_by_key(beam_candidates, 'candidate_id', ops='sum')
                all_candidates.extend(beam_candidates)
            all_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:self.m_candidates] # get top 2*m candidates
            # normalize scores
            total_score = sum([a['score'] for a in all_candidates])
            for i in range(len(all_candidates)):
                all_candidates[i]['score'] /= total_score
            # print(f"Loop {i_loop+1}: Top candidates for next step:")
            # for a in all_candidates:
            #     print(ent2name[id2ent[a['candidate_id']]], a['score'])
            new_info = []
            for candidate in all_candidates:
                new_info.append((candidate['entity_id'], candidate['relation'], candidate['candidate_id']))
            info.extend(new_info)
            
            
            ok, result = self.check_answer_sufficiency(info, self.query, self.model_args)
            if ok:
                # return self.generate_answer(info, self.query, self.model_args)
                print("The answers are: ", result)
                self.result = result
                break
            else:
                beams = []
                for a in all_candidates:
                    beams.append({"entity_id": a['candidate_id'], "score": a['score']})
                print("Not yet found the answer, continue searching...")
                i_loop += 1
            if i_loop > self.max_depth:
                print("Cannot answer the question with current information.")
                self.result = []
                break
            print("Info:", info)
            np_info = np.array(info)
            all_cands = np.concatenate([np_info[:,0], np_info[:,2]])
            self.cands = np.unique(all_cands).tolist()
            # break
                
                # for a in all_candidates:
                #     print(ent2name[id2ent[a['candidate_id']]], a['score'])

                # refine candidates with LLM
                

                
                    
                    
                    
        


if __name__ == "__main__":
    print(clean_relations(extract_relation_prompt % (3,3),23, ['relation1','relation2','relation3','relation4']))
