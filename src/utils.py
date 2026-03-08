import torch


def fuse_mean(scores: torch.Tensor):
    ls = scores.unbind(dim=0)
    res = torch.zeros(1, scores.shape[1])
    n = scores.shape[0]
    for l in ls:
        res += l / n
    return res


def fuse_rrf(scores: torch.Tensor, k=5):
    ls = scores.unbind(dim=0)
    dict_scores = {}
    for l in ls:
        _, indices = torch.sort(l, descending=True)
        for i in range(scores.shape[1]):
            if indices[i].item() not in dict_scores:
                dict_scores[indices[i].item()] = []
            dict_scores[indices[i].item()].append(i + 1)

    for key in dict_scores:
        dict_scores[key] = sum([1 / (k + rank) for rank in dict_scores[key]])
    scores = torch.tensor([list(dict_scores.values())])
    return scores


def extract_numbers(t):
    """Recursively extract all numbers from nested tuples into a flat list."""
    numbers = []
    if isinstance(t, int):
        numbers.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            numbers.extend(extract_numbers(item))
    return numbers


def extract_strings(t):
    """Recursively extract all strings from nested tuples into a flat list."""
    strings = []
    if isinstance(t, str):
        strings.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            strings.extend(extract_strings(item))
    return strings


def flatten_nested_tuple(input: tuple) -> list:
    """Recursively flatten a nested tuple into a flat list."""
    flat_list = []
    if isinstance(input, (tuple, list)):
        for item in input:
            flat_list.extend(flatten_nested_tuple(item))
    else:
        flat_list.append(input)
    return flat_list


def get_top_k(scores: torch.Tensor, k: int, fuse_function=fuse_mean):
    scores = fuse_function(scores)
    scores, sorted_indices = torch.topk(scores, k, largest=True, dim=-1)
    return sorted_indices[0].tolist()
