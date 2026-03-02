import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def normalize_answer(s):
    """
    Normalizes a string by converting to lowercase, removing underscores,
    and stripping extra whitespace.
    """
    if s is None:
        return ""
    # Replace underscores with spaces, lowercase, and strip whitespace
    return str(s).replace("_", " ").lower().strip()

def calculate_hit_at_k(ground_truth_list, predictions, k=1):
    """
    Calculates the Hit@K metric where each query has multiple valid ground truth answers.
    Normalization is applied to handle formatting differences (e.g., Barack Obama vs Barack_Obama).
    
    Hit@K is 1 if ANY of the ground truth answers for a query is within the 
    top-K predicted candidates, otherwise 0.
    """
    hits = 0
    for gts, preds in zip(ground_truth_list, predictions):
        # Normalize top K predictions
        normalized_top_k = [normalize_answer(p) for p in preds[:k]]
        # Normalize all valid ground truths for this query
        normalized_gts = [normalize_answer(gt) for gt in gts]
        
        # Check if there is any intersection
        if any(gt in normalized_top_k for gt in normalized_gts):
            hits += 1
    
    return hits / len(ground_truth_list) if len(ground_truth_list) > 0 else 0.0

def evaluate_predictions(ground_truth_list, predictions):
    """
    Runs evaluation for Accuracy, F1, Hit@1, and Hit@3.
    Applies string normalization to ground truths and predictions.
    """
    
    is_correct = []
    for gts, preds in zip(ground_truth_list, predictions):
        top_1 = normalize_answer(preds[0]) if preds else ""
        # gt = normalize_answer(gts[0]) if gts else ""
        normalized_gts = [normalize_answer(gt) for gt in gts]
        
        # Check if the top prediction matches any normalized ground truth
        is_correct.append(1 if top_1 in normalized_gts else 0)
    
    # 1. Accuracy
    accuracy = sum(is_correct) / len(is_correct) if is_correct else 0.0
    
    # 2. Hit@1
    hit_1 = calculate_hit_at_k(ground_truth_list, predictions, k=1)
    
    # 3. Hit@3
    hit_3 = calculate_hit_at_k(ground_truth_list, predictions, k=3)
    
    # F1-score calculation (using the binary correctness vector)
    f1 = f1_score([1] * len(is_correct), is_correct, zero_division=0)
    
    # Print Results
    # print("--- Evaluation Results (Normalized Multi-GT) ---")
    # print(f"Accuracy (Top-1): {accuracy:.4f}")
    # print(f"F1 Score (Top-1): {f1:.4f}")
    # print(f"Hit@1:           {hit_1:.4f}")
    # print(f"Hit@3:           {hit_3:.4f}")
    # print("-------------------------------------------------")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "hit@1": hit_1,
        "hit@3": hit_3
    }

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10

if __name__ == "__main__":
    # Example Data showing formatting differences
    gt_data = [
        ["Barack Obama", "B. Obama"],        # Underscore in prediction should match
        ["New York City", "NYC"],           # Case sensitivity check
        ["cherry"], 
        ["Saint_Petersburg", "St Petersburg"] 
    ]
    
    pred_data = [
        ["Barack_Obama", "Obama"],           # Hit@1 via normalization (replaces _ with space)
        ["new york city", "Jersey"],         # Hit@1 via normalization (lowercase)
        ["Berry", "Cherry", "Apple"],        # Hit@3 via normalization
        ["St. Petersburg", "St Petersburg"]   # Hit@3 (exact match with the second GT)
    ]
    
    evaluate_predictions(gt_data, pred_data)