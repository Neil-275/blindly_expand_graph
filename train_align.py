from model_2 import *
import torch
import pickle as pkl
import wandb
import random
from loader2 import DataLoader2
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm
from utils import cal_performance




def loadModel(filePath, model):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device('cuda:0'))
        if checkpoint.get('model_state_dict') is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        return model
        # re-build optimizter
        # self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)

class Trainer:
    def __init__(self, gnn_model, projector, train_loader, val_loader, args):
        self.gnn_model = gnn_model
        self.projector = projector
        
        self.args = args
        self.t_time = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer, self.scheduler = self.setup_training(train_loader)

    def setup_training(self, train_loader):
        """
        Initializes the optimizer and scheduler for Phase 2.
        """
        # Use weight decay to regularize the node embeddings


        # Create parameter groups
        param_groups = [
            {'params': self.gnn_model.parameters(), 'lr': self.args.gnn_lr},
            {'params': self.projector.parameters(), 'lr': self.args.projector_lr}
        ]

        # Create optimizer with different learning rates
        optimizer = Adam(param_groups, weight_decay=1e-2)
        
        # OneCycleLR is great for ranking tasks as it helps escape local minima
        total_steps = len(train_loader) * self.args.epochs
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=self.args.gnn_lr * 10, 
            total_steps=total_steps,
            pct_start=0.1, 
            anneal_strategy='cos'
        )
    
        return optimizer, scheduler
    
    def prepareData(self, batch_data):
        subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
        subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
        subs = subs.cuda().flatten()
        rels = rels.cuda().flatten()
        objs = objs.cuda()
        return subs, rels, objs, subgraph_data

    def train_batch(self):        
        # ov_str = ""
        epoch_loss = 0
        reach_tails_list = []
        t_time = time.time()
        self.gnn_model.train()
        k = 0
        
        for batch_data in tqdm(self.train_loader, ncols=50, leave=False):                      
            # prepare data    
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            # print(subs.max())
            # forward
            self.gnn_model.zero_grad()
            scores = self.gnn_model(subs, rels, subgraph_data, self.projector)
            print(scores.shape)
            k += 1
            if k == 2:
                break
        #     # loss calculation
        #     pos_scores = scores[[torch.arange(len(scores)).cuda(), objs.flatten()]]
        #     max_n = torch.max(scores, 1, keepdim=True)[0]
        #     loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 

        #     # loss backward
        #     loss.backward()
        #     self.optimizer.step()

        #     # avoid NaN
        #     # for p in self.model.parameters():
        #     #     X = p.data.clone()
        #     #     flag = X != X
        #     #     X[flag] = np.random.random()
        #     #     p.data.copy_(X)

        #     # cover tail entity or not
        #     reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
        #     reach_tails_list += reach_tails
        #     epoch_loss += loss.item()
        #     break
        # self.t_time += time.time() - t_time
        
        # # evaluate on val/test set
        # valid_mrr, out_str = self.evaluate(eval_train=True)    
        # self.scheduler.step(valid_mrr)
        print("goodbye")
        
        # return valid_mrr, out_str
    
    @torch.no_grad()
    def evaluate(self, eval_train=False, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False):
        self.model.eval()
        i_time = time.time()
        
        # eval on train set
        if eval_train:
            print("evaluating on train set...")
            ranking = []
            stop = 0
            train_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='train')  # keep on GPU
                
                # calculate rank - train mode has different obj format, all on GPU
                objs = objs.flatten()  # flatten to get single target indices
                batch_size = scores.size(0)
                
                # Create ranking for each query in the batch on GPU
                for i in range(batch_size):
                    # Get filter for this specific query
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1
                    
                    # Calculate rank for single target entity on GPU
                    target_score = scores[i, objs[i]]
                    # Count how many entities score higher (excluding filtered entities)
                    higher_scores = scores[i] > target_score
                    higher_scores = higher_scores & (1 - filt_1hot).bool()
                    rank = torch.sum(higher_scores).item() + 1
                    ranking.append(rank)
                    
                    if mean_rank:
                        mean_rank_list.append(rank)

                # cover tails or not - adapted for train mode, on GPU
                for i in range(batch_size):
                    target_score = scores[i, objs[i]]
                    reach_tail = 1 if target_score.item() == 0 else 0
                    train_reach_tails_list.append(reach_tail)
                    
                stop += 1
                if stop == 15:
                    break

            ranking = np.array(ranking)
            tr_mrr, tr_h1, tr_h10 = cal_performance(ranking)
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Train set:\n', rank_CR)
        
            # save mean rank
            if mean_rank: self.mean_rank_dict['train'] = copy.deepcopy(mean_rank_list)
                
        else:
            tr_mrr, tr_h1, tr_h10 = -1, -1, -1
        
        # eval on val set
        if eval_val:
            print("evaluating on val set...")
            ranking = []
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='valid')  # keep on GPU

                # calculate rank on GPU
                batch_size = scores.size(0)
                filters = []
                for i in range(batch_size):
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1
                    filters.append(filt_1hot)
                filters = torch.stack(filters)  # [batch_size, n_ent]
                
                # Calculate ranks on GPU using cal_ranks equivalent
                ranks = []
                for i in range(batch_size):
                    # Get target entities for this query (multi-hot format)
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    query_ranks = []
                    
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        # Count entities with higher scores (excluding filtered)
                        higher_scores = scores[i] > target_score
                        higher_scores = higher_scores & (1 - filters[i]).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)
                    
                    # Use minimum rank (best) for this query
                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)

                ranking += ranks

                # cover tails or not - on GPU
                for i in range(batch_size):
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        reach_tail = 1 if target_score.item() == 0 else 0
                        val_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)
            # print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Val set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['val'] = copy.deepcopy(mean_rank_list)
                
        else:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        
        # eval on test set
        if eval_test:
            print("evaluating on test set...")
            ranking = []
            test_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                # prepare data            
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                # forward
                scores = self.model(subs, rels, subgraph_data, mode='test')  # keep on GPU

                # calculate rank on GPU
                batch_size = scores.size(0)
                filters = []
                for i in range(batch_size):
                    filt = self.loader.filters[(subs[i].item(), rels[i].item())]
                    filt_1hot = torch.zeros(self.n_ent, device=scores.device)
                    filt_1hot[list(filt)] = 1
                    filters.append(filt_1hot)
                filters = torch.stack(filters)  # [batch_size, n_ent]
                
                # Calculate ranks on GPU
                ranks = []
                for i in range(batch_size):
                    # Get target entities for this query (multi-hot format)
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    query_ranks = []
                    
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        # Count entities with higher scores (excluding filtered)
                        higher_scores = scores[i] > target_score
                        higher_scores = higher_scores & (1 - filters[i]).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)
                    
                    # Use all ranks for this query
                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)

                ranking += ranks

                # cover tails or not - on GPU
                for i in range(batch_size):
                    target_entities = torch.nonzero(objs[i]).squeeze(-1)
                    for target_ent in target_entities:
                        target_score = scores[i, target_ent]
                        reach_tail = 1 if target_score.item() == 0 else 0
                        test_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            # print(f'[test] covering tail ratio: {len(test_reach_tails_list)}, {1 - sum(test_reach_tails_list) / len(test_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Test set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['test'] = copy.deepcopy(mean_rank_list)
            
        else:
            t_mrr, t_h1, t_h10 = -1, -1, -1
            
        i_time = time.time() - i_time
        out_str = '[TRAIN] MRR:%.4f H@1:%.4f H@10:%.4f\t [VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(tr_mrr, tr_h1, tr_h10, v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        return v_mrr, out_str

class GNN_config:
    n_layer = 8
    hidden_dim = 64
    attn_dim = 4
    dropout = 0.3
    n_ent = 1024 # subgraph size
    shortcut = False
    readout = 'linear'
    concatHidden = True
    initializer = 'binary'
    llm_description_emb_path = "data_for_CL/llm_description_aligned_emb.pkl"
    with open(llm_description_emb_path, "rb") as f:
        llm_description_aligned_emb = pkl.load(f)
    llm_emb = list(llm_description_aligned_emb.values())
    llm_emb = torch.stack(llm_emb, dim=0)
    del(llm_description_aligned_emb)

class Projector_config:
    n_layers = 8
    in_dim = 4096
    hidden_dims = [512, 256]
    out_dim = 64

class Training_args:
    gnn_lr = 1e-4
    projector_lr = 1e-3
    epochs = 50

if __name__ == "__main__":
    with open("for_finetuning.pkl", "rb") as f:
        data = pkl.load(f)
    
    train_len = int(len(data) * 0.8)
    train_data = data[:train_len]
    val_data = data[train_len:]
    train_loader = DataLoader2(train_data, mode='train')
    train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=True, collate_fn=train_loader.collate_fn)
    val_loader = DataLoader2(val_data, mode='eval')
    val_dataloader = DataLoader(val_loader, batch_size=32, shuffle=False, collate_fn=val_loader.collate_fn)


    gnn_model = GNN_auto(GNN_config, train_loader)
    pretrain_model_path = "weights/topk_0.1_layer_8_ValMRR_0.437.pt"
    pretrain_gnn_model = loadModel(pretrain_model_path, gnn_model)

    projector_model = Projector(Projector_config.in_dim, Projector_config.hidden_dims, Projector_config.out_dim, Projector_config.n_layers)
    pretrain_projector_path = "weights/projectors/best_projector_HNM.pt"
    pretrain_projector_model = loadModel(pretrain_projector_path, projector_model)
    print("Models loaded successfully.")

    trainer = Trainer(pretrain_gnn_model, pretrain_projector_model, train_dataloader, val_dataloader, Training_args)
    trainer.train_batch()
        
