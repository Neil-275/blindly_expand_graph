from difflib import SequenceMatcher
import re
from prompt_list2 import decision_prompt, entities_pruning_prompt, GoG_answer_prompt
from prompt_list import extract_relation_prompt
from data_utils import id2ent, id2rel, ent2id, ent2name, name2ent, rel2id

def map_to_most_similar(returned_string, input_list, threshold=0.0):
    if not returned_string or not input_list:
        return None
    
    # Clean up the returned string (remove extra whitespace, newlines)
    returned_string = returned_string.strip()
    
    best_match = None
    best_score = 0.0
    
    for candidate in input_list:
        # Calculate similarity using multiple methods and take the maximum
        
        # Method 1: SequenceMatcher for overall similarity
        seq_score = SequenceMatcher(None, returned_string.lower(), candidate.lower()).ratio()
        
        # Method 2: Check if returned string is a prefix/suffix of candidate (common with truncation)
        prefix_score = 0.0
        if candidate.lower().startswith(returned_string.lower()):
            prefix_score = len(returned_string) / len(candidate)
        elif candidate.lower().endswith(returned_string.lower()):
            prefix_score = len(returned_string) / len(candidate)
        
        # Method 3: Check if candidate contains the returned string
        contains_score = 0.0
        if returned_string.lower() in candidate.lower():
            contains_score = len(returned_string) / len(candidate)
        
        # Method 4: Jaccard similarity for word-level matching
        returned_words = set(returned_string.lower().split())
        candidate_words = set(candidate.lower().split())
        if returned_words or candidate_words:
            jaccard_score = len(returned_words & candidate_words) / len(returned_words | candidate_words)
        else:
            jaccard_score = 0.0
        
        # Take the maximum of all similarity scores
        current_score = max(seq_score, prefix_score, contains_score, jaccard_score)
        
        if current_score > best_score:
            best_score = current_score
            best_match = candidate
    
    # Return best match only if it meets the threshold
    return best_match if best_score >= threshold else None

def map_to_most_similar_list(returned_strings, input_list, threshold=0.4):

    return [map_to_most_similar(s, input_list, threshold) for s in returned_strings] 

def construct_relation_prune_prompt(question, entity_name, total_relations, k):
    
    for i in range(len(total_relations)):
    #     if total_relations[i] in id2rel:
            total_relations[i] = total_relations[i][1:]
    return extract_relation_prompt % (k, k) + question[0] + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '\n'.join(total_relations) + "\nA: "

def construct_decision_prompt(question, entity_name, relations):
    q = question[0] if isinstance(question, list) else question
    rel_str = "\n".join(relations)
    prompt = decision_prompt
    prompt = prompt.replace("{{question}}", q)
    prompt = prompt.replace("{{current_entity}}", entity_name)
    prompt = prompt.replace("{{reasoning_history}}", "")
    prompt = prompt.replace("{{context}}", rel_str)
    return prompt + '\nA: '

def extract_decision(string):
    # print("LLM decision response:", string)
    pattern = r'^(Yes|No)'
    match = re.search(pattern, string, re.IGNORECASE)
    if match:
        # print("Extracted decision:", match.group(1))
        return match.group(1)
    else:
        return "No clear decision found"

def clean_relations(string, entity_id):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation[0]=="+":
            relations.append({"entity_id": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity_id": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations
