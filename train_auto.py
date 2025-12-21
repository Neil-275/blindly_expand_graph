import os
import argparse
import torch
import time
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from torch.utils.data import DataLoader as TorchDataLoader
from utils import *
# from PPR_sampler import pprSampler
from  expand_subgraph import ExpandSubgraph 

parser = argparse.ArgumentParser(description="Parser for the one-shot-subgraph framework")
parser.add_argument('--data_path', type=str, default='knowledge_graph/KG_data/FB15k-237-betae')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--k', type=float, default=6) # number of sampled nodes (for a subgraph)
parser.add_argument('--depth', type=float, default=8) # number of sampled edges (for a subgraph)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fact_ratio', type=float, default=0.75)
parser.add_argument('--val_num', type=int, default=-1) # how many triples are used as the validate set
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--add_manual_edges', action='store_true')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--not_shuffle_train', action='store_true')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(8, args.cpu))
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, dataset)):
        os.makedirs(os.path.join(results_dir, dataset))

    opts = args
    time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    opts.perf_file = os.path.join(results_dir,  dataset + '\\' + time + '.txt')
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    print('==> gpu:', gpu)
    opts.n_batch = opts.n_tbatch = int(args.batchsize)
    with open(opts.perf_file, 'a+') as f:
        f.write(str(opts))
    
    loader = DataLoader(args, mode='train')
    val_loader = DataLoader(args, mode='valid')
    test_loader = DataLoader(args, mode='test')
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    
    # build ppr sampler here
    # number of sampled entities
    print(f'==> #branching factor: {args.k}, #depth: {args.depth}')
    
    # sampler for training
    train_graph_homo = list(set([(h,t) for (h,r,t) in loader.train_graph]))
    # fact_data = np.concatenate([np.array(loader.fact_data, dtype=np.int32), loader.idd_data], 0, dtype=np.int32)
    train_graph = np.array(loader.train_graph, dtype=np.int32)
    train_sampler = ExpandSubgraph(args.n_ent, args.n_rel,train_graph_homo,train_graph,
                                   args=args, k=args.k, depth=args.depth)


    # sampler for
    val_graph_homo = list(set([(h,t) for (h,r,t) in loader.val_graph]))
    # test_data = np.concatenate([np.array(test_data, dtype=np.int32), loader.idd_data], 0, dtype=np.int32)
    val_graph = np.array(loader.val_graph, dtype=np.int32)
    val_sampler = ExpandSubgraph(args.n_ent, args.n_rel,val_graph_homo,val_graph,
                                   args=args)


    # sampler for validation
    test_graph_homo = list(set([(h,t) for (h,r,t) in loader.test_graph]))
    # test_data = np.concatenate([np.array(test_data, dtype=np.int32), loader.idd_data], 0, dtype=np.int32)
    test_graph = np.array(loader.test_graph, dtype=np.int32)
    test_sampler = ExpandSubgraph(args.n_ent, args.n_rel,test_graph_homo,test_graph,
                                    args=args)
                                    

        
    # add sampler to the data loaders
    loader.addSampler(train_sampler)
    val_loader.addSampler(val_sampler)
    test_loader.addSampler(test_sampler)
    # check all output paths
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{args.data_path}/saveModel/')
            
    def run_model(params):       
        print('==> start training...')    
        # print(params)
        args.lr = params['lr']
        args.decay_rate = params['decay_rate']
        args.lamb = params['lamb']
        args.hidden_dim = params['hidden_dim']
        args.attn_dim = params['attn_dim']
        args.n_layer = args.layer = params['n_layer']
        args.dropout = params['dropout']
        args.act = params['act']
        args.initializer = params['initializer']
        args.concatHidden = params['concatHidden']
        args.shortcut = params['shortcut']
        args.readout = params['readout']
        
        # build model
        model = BaseModel(args, loaders=(loader, val_loader, test_loader), samplers=(train_sampler, val_sampler, test_sampler))
        
        # load pretrained weight
        if args.weight != '': 
            model.loadModel(args.weight)
            
        # only do evaluation, and then exit
        if args.only_eval:
            valid_mrr, out_str = model.evaluate(verbose=True, rank_CR=False)
            print(out_str)
            exit()

        # training
        best_mrr, best_test_mrr, bearing = 0, 0, 0
        for epoch in range(args.epoch):
            mrr, out_str = model.train_batch()
            
            with open(opts.perf_file, 'a+') as f:
                f.write(out_str)
                
            if mrr > best_mrr:
                best_mrr = mrr
                best_str = out_str
                print(str(epoch) + '\t' + best_str)
                bearing = 0
                
                # save model weight (by default)
                BestMetricStr = f'ValMRR_{str(mrr)[:5]}'
                model.saveModelToFiles(args, BestMetricStr, deleteLastFile=False)
            else:
                bearing += 1
                
            # early stop
            if bearing >= 20: 
                print(f'early stopping at {epoch+1} epoch.')
                break
        
        print(best_str)
        return best_mrr
    
    # NOTE: best config
    if dataset == 'WN18RR':
        # [VALID] MRR:0.5690 H@1:0.5170 H@10:0.6663        [TEST] MRR:0.5678 H@1:0.5140 H@10:0.6662
        params = {'lr': 0.0001, 'hidden_dim': 256, 'attn_dim': 8, 'n_layer': 8, 'act': 'idd', 'initializer': 'relation', 'concatHidden': False, 'shortcut': True, 'readout': 'multiply', 'decay_rate': 0.8662400068095666, 'lamb': 0.00039154537550520227, 'dropout': 0.004323645605227445}
    elif dataset == 'nell':
        # [VALID] MRR:0.5051 H@1:0.4355 H@10:0.6133        [TEST] MRR:0.5472 H@1:0.4847 H@10:0.6508
        params = {'lr': 0.0011, 'hidden_dim': 128, 'attn_dim': 64, 'n_layer': 8, 'act': 'relu', 'initializer': 'relation', 'concatHidden': False, 'shortcut': False, 'readout': 'linear', 'decay_rate': 0.9938, 'lamb': 0.000089, 'dropout': 0.0193}
    elif dataset == 'YAGO':
        # [VALID] MRR:0.6117 H@1:0.5477 H@10:0.7273        [TEST] MRR:0.6064 H@1:0.5403 H@10:0.7218 
        params = {'lr': 0.001, 'hidden_dim': 64, 'attn_dim': 2, 'n_layer': 8, 'act': 'relu', 'initializer': 'binary', 'concatHidden': True, 'shortcut': False, 'readout': 'linear', 'decay_rate': 0.9429713470775948, 'lamb': 0.000946516892415447, 'dropout': 0.19456805575101324}
    elif dataset == 'FB15k-237-betae':
        # [VALID] MRR:0.6117 H@1:0.5477 H@10:0.7273        [TEST] MRR:0.6064 H@1:0.5403 H@10:0.7218 
        params = {'lr': 0.001, 'hidden_dim': 64, 'attn_dim': 2, 'n_layer': 8, 'act': 'relu', 'initializer': 'binary', 'concatHidden': True, 'shortcut': False, 'readout': 'linear', 'decay_rate': 0.9429713470775948, 'lamb': 0.000946516892415447, 'dropout': 0.19456805575101324}
    else:
        exit()
    # print("175")
    run_model(params)