import time
import openai
import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import os

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

def extract_notations(t):
    """Recursively extract all notations (entities and relations) from nested tuples into a flat list."""
    notations = []
    t = str(t)
    for c in t:
        if c.isalpha():
            notations.append(c)
    return notations

def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""

def run_llm(prompt, system_prompt, max_tokens=500, temperature=0.5, engine="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # while True:
    result = []
    try:
        response = openai.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0
        )
        result = response.choices[0].message.content
        # break
    except Exception as e:
        print(f"openai error, retry: {e}")
        time.sleep(2)
    print("_______end openai")
    return result



def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)

def cal_ranks_mean(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    batch_idx, ent_idx = np.nonzero(ranks)
    mean_rank = [[] for i in range(int(ranks.shape[0]))]
    for i in range(len(batch_idx)):
        x, y = batch_idx[i], ent_idx[i]
        rank = ranks[x,y]
        mean_rank[x].append(rank)
    return mean_rank

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            #proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0,len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0,len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i
        
def calculate_statistics(data):
    """
    Calculates descriptive statistics for a given array-like object.

    Args:
        data (array-like): A list or NumPy array of numerical data.

    Returns:
        dict: A dictionary containing the statistics, or a dictionary of Nones if the input is empty.
    """
    # Ensure data is a numpy array
    data_arr = np.asarray(data)

    # Handle empty array
    if data_arr.size == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std_dev': None,
            'min': None,
            'max': None,
            '25th_percentile': None,
            '75th_percentile': None
        }

    stats = {
        'count': data_arr.size,
        'mean': np.mean(data_arr),
        'median': np.median(data_arr),
        'std_dev': np.std(data_arr),
        'min': np.min(data_arr),
        'max': np.max(data_arr),
        '25th_percentile': np.percentile(data_arr, 25),
        '75th_percentile': np.percentile(data_arr, 75)
    }
    return stats