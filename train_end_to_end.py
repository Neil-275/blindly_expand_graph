from model_2 import *
import torch
import pickle as pkl
import wandb
import random
import numpy as np
from loader2 import DataLoader2
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()
from utils import cal_performance
from torch_scatter import scatter_max, scatter_add
import torch.nn.functional as F


def get_latest_checkpoint():
    """Find and return the path to the latest checkpoint.
    
    Returns:
        Path to the latest checkpoint or None if no checkpoint exists
    """
    if not os.path.exists('saveModels'):
        return None
    
    checkpoint_files = [f for f in os.listdir('saveModels') if f.startswith('topk_') and f.endswith('.pt')]
    if not checkpoint_files:
        return None
    
    # Get the most recently modified checkpoint
    latest_checkpoint = max(
        checkpoint_files,
        key=lambda f: os.path.getmtime(os.path.join('saveModels', f))
    )
    return os.path.join('saveModels', latest_checkpoint)


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
    def __init__(self, gnn_model, projector, train_loader, val_loader, val_train_loader, args):
        self.gnn_model = gnn_model.to(args.device)
        self.projector = projector.to(args.device)
        self.temperature = args.temperature 
        # gnn_model._setup_readout() 
        self.args = args
        self.t_time = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_train_loader = val_train_loader
        # print(len(val_loader))
        self.optimizer_gnn, self.optimizer_proj, self.scheduler_gnn, self.scheduler_proj = self.setup_training(train_loader)

    def setup_training(self, train_loader):
        """
        Initializes the optimizer and scheduler for Phase 2.
        """
        # Use weight decay to regularize the node embeddings


        # Create parameter groups
        # param_groups = [
        #     {'params': self.gnn_model.parameters(), 'lr': self.args.gnn_lr},
        #     {'params': self.projector.parameters(), 'lr': self.args.projector_lr}
        # ]

        # Create optimizer with different learning rates
        optimizer_gnn = Adam(self.gnn_model.parameters(), lr=self.args.gnn_lr, weight_decay=1e-2)
        optimizer_proj = Adam(self.projector.parameters(), lr=self.args.projector_lr, weight_decay=1e-2)
        # ReduceLROnPlateau reduces LR when validation metric plateaus
        scheduler_gnn = ReduceLROnPlateau(
            optimizer_gnn, 
            mode='max', 
            factor=0.5, 
            patience=2, 
            min_lr=self.args.gnn_lr / 20
        )
        scheduler_proj = ReduceLROnPlateau(
            optimizer_proj, 
            mode='max', 
            factor=0.5, 
            patience=2, 
            min_lr=self.args.projector_lr / 20
        )
    
        return optimizer_gnn, optimizer_proj, scheduler_gnn, scheduler_proj 
    
    def prepareData(self, batch_data):
        subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
        subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, query_tail_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
        subs = subs.cuda().flatten()
        rels = rels.cuda().flatten()
        objs = objs.cuda()
        return subs, rels, objs, subgraph_data


    def compute_local_loss(self, scores, batch_idxs, query_tail_idxs, query_head_idxs):
        # 1. Mask the Head: We set the head score to a very large negative value.
        # This ensures the head is never the 'max_n' and contributes 0 to the LogSumExp.
        masked_scores = scores.clone()
        # print("Before masking, head scores:", scores[query_head_idxs])
        # masked_scores[query_head_idxs] = -1e10 

        # 2. Positive scores (the actual tails we want to find)
        pos_scores = scores[query_tail_idxs]
        
        # 3. Denominator (LogSumExp) calculation over the rest of the nodes
        max_n, _ = scatter_max(masked_scores, batch_idxs, dim=0)
        max_n_broadcasted = max_n[batch_idxs]
        exp_shifted = torch.exp(masked_scores - max_n_broadcasted)
        sum_exp = scatter_add(exp_shifted, batch_idxs, dim=0)
        log_sum_exp = torch.log(sum_exp + 1e-10)
        
        # Formula: -Score(pos) + LogSumExp(all_nodes_except_head)
        per_query_loss = -pos_scores + max_n + log_sum_exp
        
        return torch.sum(per_query_loss)

    def saveModelToFiles(self, args, best_metric, epoch, best_mrr, deleteLastFiles=True):
        # if args.val_num == -1:
        #     savePath = f'/saveModel/topk_{best_metric}.pt'
        # else:
        if deleteLastFiles:
            previous_files = [f for f in os.listdir('saveModels') if f.startswith('topk_')]
            for pf in previous_files:
                os.remove(os.path.join('saveModels', pf))
        savePath = f'saveModels/topk_{best_metric}.pt'
            
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'gnn_model': self.gnn_model.state_dict(),
                'projector': self.projector.state_dict(),
                'optimizer_gnn_state_dict': self.optimizer_gnn.state_dict(),
                'optimizer_proj_state_dict': self.optimizer_proj.state_dict(),
                'scheduler_gnn_state_dict': self.scheduler_gnn.state_dict(),
                'scheduler_proj_state_dict': self.scheduler_proj.state_dict(),
                'best_mrr': best_mrr,
                'epoch': epoch,
                }, savePath)
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary with 'epoch', 'best_mrr', and other training state
        """
        print(f'Loading checkpoint from {checkpoint_path}')
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))
        
        # Load model states
        if checkpoint.get('gnn_model') is not None:
            self.gnn_model.load_state_dict(checkpoint['gnn_model'])

        if checkpoint.get('projector') is not None:
            self.projector.load_state_dict(checkpoint['projector'])
        if checkpoint.get('optimizer_gnn_state_dict') is not None:
            self.optimizer_gnn.load_state_dict(checkpoint['optimizer_gnn_state_dict'])
        if checkpoint.get('optimizer_proj_state_dict') is not None:
            self.optimizer_proj.load_state_dict(checkpoint['optimizer_proj_state_dict'])
        
        # Load scheduler state if available
        if checkpoint.get('scheduler_gnn_state_dict') is not None:
            self.scheduler_gnn.load_state_dict(checkpoint['scheduler_gnn_state_dict'])
        if checkpoint.get('scheduler_proj_state_dict') is not None:
            self.scheduler_proj.load_state_dict(checkpoint['scheduler_proj_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        best_mrr = checkpoint.get('best_mrr', 0)
        
        print(f'Checkpoint loaded: epoch={epoch}, best_mrr={best_mrr}')
        return {'epoch': epoch, 'best_mrr': best_mrr}
        

    def train_batch(self, epoch_idx):        
        # ov_str = ""
        epoch_loss = 0
        reach_tails_list = []
        t_time = time.time()
        
        k = 0
        llm_emb = GNN_config.llm_emb.cuda()
        train_ver = 10
        self.gnn_model.train()
        self.projector.train()

        # torch.cuda.memory._record_memory_history(max_entries=100000)
        start_time = time.time()
        pbar = tqdm(self.train_loader, ncols=50, desc=f"Epoch {epoch_idx}")
        stop = 2
        for i, batch_data in enumerate(pbar):                      
        # prepare data    
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            batch_idxs, _, query_sub_idxs, query_tail_idxs, _, batch_sampled_edges = subgraph_data
            batch_idxs = batch_idxs.cuda()
            query_tail_idxs = query_tail_idxs.cuda()
            query_sub_idxs = query_sub_idxs.cuda()
            # forward
            # print("shape check:", subs.shape, rels.shape, objs.shape)
            self.optimizer_gnn.zero_grad()
            self.optimizer_proj.zero_grad()
            # scores = self.gnn_model(subs, rels, subgraph_data, self.projector)
            scores = self.gnn_model(subs, rels, subgraph_data, self.projector, use_llm=True)  # keep on GPU
            assert scores.shape[0] == batch_idxs.shape[0]
            loss_lp = self.compute_local_loss(scores, batch_idxs, query_tail_idxs, query_sub_idxs)
 
            #######  Contrastive loss ##########
            all_rels = torch.cat([batch_sampled_edges[:,1], rels], dim = 0)
            unique_rels = torch.unique(all_rels)

            contrastive_loss = torch.tensor(0.0, device='cuda')

            for i_layer in range(GNN_config.active_layer):
                # forward
                # random select a version for each unique relation
                ver_idx = torch.randint(0,train_ver,(unique_rels.shape[0],)).cuda()
                llm_emb_rel = llm_emb[unique_rels, ver_idx, :]

                proj_emb_rel = self.projector.models[i_layer](llm_emb_rel) 
                gnn_emb_rel = self.gnn_model.gnn_layers[i_layer].rela_embed.weight.detach()

                proj_emb_rel = F.normalize(proj_emb_rel, p=2, dim=1)
                gnn_emb_rel = F.normalize(gnn_emb_rel, p=2, dim=1)
                
                # print("proj_emb_rel shape:", proj_emb_rel.shape)
                # print("gnn_emb_rel shape:", gnn_emb_rel.shape)
                logits = torch.matmul(proj_emb_rel, gnn_emb_rel.T) / self.temperature
                # print("logits shape:", logits.shape)
                layer_cont_loss = F.cross_entropy(logits, unique_rels)
                contrastive_loss = contrastive_loss + layer_cont_loss
                
                break
            break

            total_loss = loss_lp + contrastive_loss
            total_loss.backward()

            self.optimizer_gnn.step()
            self.optimizer_proj.step()
            # cover tail entity or not
            pos_scores = scores[query_tail_idxs]
            reach_tails = (pos_scores != 0).detach().int().cpu().tolist()
            reach_tails_list += reach_tails
            epoch_loss += total_loss.item()
            # print("\nLoss with head masking: ", loss_lp.item())
            if i % 10 == 0:
                wandb.log({
                    "batch/total_loss": total_loss.item(),
                    "batch/lp_loss": loss_lp.item(),
                    "batch/contrastive_loss": contrastive_loss.item(),
                    "batch/gnn_lr": self.optimizer_gnn.param_groups[0]['lr'],
                    "batch/proj_lr": self.optimizer_proj.param_groups[0]['lr']
                })
        self.t_time += time.time() - t_time
        
        # evaluate on val/test set
        valid_mrr, metrics = self.evaluate(eval_train=True)    
        wandb.log({
            "epoch/avg_loss": epoch_loss / len(self.train_loader),
            "epoch/train_mrr": metrics['tr_mrr'],
            "epoch/train_h1": metrics['tr_h1'],
            "epoch/train_h2": metrics['tr_h2'],
            "epoch/train_h3": metrics['tr_h3'],
            "epoch/train_h10": metrics['tr_h10'],
            "epoch/tr_head_ratio": metrics['tr_head_ratio'],
            "epoch/val_mrr": metrics['v_mrr'],
            "epoch/val_h1": metrics['v_h1'],
            "epoch/val_h2": metrics['v_h2'],
            "epoch/val_h3": metrics['v_h3'],
            "epoch/val_h10": metrics['v_h10'],
            "epoch/v_head_ratio": metrics['v_head_ratio'],
            "epoch/time": time.time() - start_time
        })
        self.scheduler_gnn.step(valid_mrr)
        self.scheduler_proj.step(valid_mrr)
        # # print("goodbye")
        
        return valid_mrr, metrics, metrics['out_str']
    
    @torch.no_grad()
    def evaluate(self, eval_train=False, eval_val=True, eval_test=False, verbose=False, rank_CR=False, mean_rank=False):
        self.gnn_model.eval()
        self.projector.eval()
        i_time = time.time()
        stop = 2
        # eval on train set
        if eval_train:
            print("evaluating on train set...")
            # print(len(self.train_loader))
            cnt = 0
            total = 0
            ranking = []
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for i, batch_data in enumerate(tqdm(self.val_train_loader, ncols=50)):      
                # prepare data            
                # print(i)
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                total += subs.shape[0]
                batch_idxs, _, query_sub_idxs, query_tail_idxs, _, _ = subgraph_data
                batch_idxs = batch_idxs.cuda()
                query_tail_idxs = query_tail_idxs.cuda()
                query_sub_idxs = query_sub_idxs.cuda()
                # forward
                # scores = self.gnn_model(subs, rels, subgraph_data, self.projector, mode='valid') 
                scores = self.gnn_model(subs, rels, subgraph_data, self.projector, mode='valid', use_llm=True)  # keep on GPU
                assert scores.shape[0] == batch_idxs.shape[0] and scores.shape[0] == query_tail_idxs.shape[0]
                # print(query_tail_idxs.shape, scores.shape, batch_idxs.shape)
                batch_size = batch_idxs.max().item() + 1

                ranks = []
                # local ranking calculation
                for i in range(batch_size):
                    subgraph_mask = (batch_idxs == i)
                    subgraph_scores = scores[subgraph_mask]
                    labels = query_tail_idxs[subgraph_mask]
                    target_entities = torch.nonzero(labels)
                    query_ranks = []

                    for target_ent in target_entities:
                        target_score = subgraph_scores[target_ent]
                        higher_scores = subgraph_scores > target_score
                        higher_scores = higher_scores & (~labels).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)

                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)

                

                # Check if head is in rank 1 of scores
                best_score = scatter_max(scores, batch_idxs, dim=0)[0]
                head_score = scores[query_sub_idxs]
                cnt += torch.sum((head_score >= best_score).int()).item()

                
                ranking += ranks

                # cover tails or not - on GPU
                # for i in range(batch_size):
                #     target_entities = torch.nonzero(objs[i]).squeeze(-1)
                #     for target_ent in target_entities:
                #         target_score = scores[i, target_ent]
                #         reach_tail = 1 if target_score.item() == 0 else 0
                #         val_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            tr_mrr, tr_h1, tr_h2, tr_h3, tr_h10 = cal_performance(ranking)
            # print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
            
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
            print(f"\nCovering head ratio in train set: {cnt}/{total} = {cnt/total}\n")
            tr_head_ratio = cnt/total
        else:
            tr_mrr, tr_h1, tr_h2, tr_h3, tr_h10 = -1, -1, -1, -1, -1
        
        
        # eval on val set
        if eval_val:
            print("evaluating on val set...")
            # print(len(self.val_loader))
            ranking = []
            cnt = 0
            total = 0
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for i, batch_data in enumerate(tqdm(self.val_loader, ncols=50)):
                # prepare data
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                total += subs.shape[0]
                batch_idxs, _, query_sub_idxs, query_tail_idxs, _, _ = subgraph_data
                batch_idxs = batch_idxs.cuda()
                query_tail_idxs = query_tail_idxs.cuda()
                query_sub_idxs = query_sub_idxs.cuda()
                # forward
                # scores = self.gnn_model(subs, rels, subgraph_data, self.projector, mode='valid')  # keep on GPU
                scores = self.gnn_model(subs, rels, subgraph_data, self.projector, mode='valid', use_llm=True)  # keep on GPU
                
                assert scores.shape[0] == batch_idxs.shape[0] and scores.shape[0] == query_tail_idxs.shape[0]
                # print(query_tail_idxs.shape, scores.shape, batch_idxs.shape)
                batch_size = batch_idxs.max().item() + 1

                
                ranks = []
                # local ranking calculation
                for i in range(batch_size):
                    subgraph_mask = (batch_idxs == i)
                    subgraph_scores = scores[subgraph_mask]
                    labels = query_tail_idxs[subgraph_mask]
                    target_entities = torch.nonzero(labels)
                    query_ranks = []

                    for target_ent in target_entities:
                        target_score = subgraph_scores[target_ent]
                        higher_scores = subgraph_scores > target_score
                        higher_scores = higher_scores & (~labels).bool()
                        rank = torch.sum(higher_scores).item() + 1
                        query_ranks.append(rank)

                    ranks.extend(query_ranks)
                    if mean_rank:
                        mean_rank_list.extend(query_ranks)
                
                ranking += ranks

                # Check if head is in rank 1 of scores
                    # subgraph_mask = (batch_idxs == i)
                    # subgraph_scores = scores[subgraph_mask]
                best_score = scatter_max(scores, batch_idxs, dim=0)[0]
                head_score = scores[query_sub_idxs]
                cnt += torch.sum((head_score >= best_score).int()).item()
                
                

                # cover tails or not - on GPU
                # for i in range(batch_size):
                #     target_entities = torch.nonzero(objs[i]).squeeze(-1)
                #     for target_ent in target_entities:
                #         target_score = scores[i, target_ent]
                #         reach_tail = 1 if target_score.item() == 0 else 0
                #         val_reach_tails_list.append(reach_tail)

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h2, v_h3, v_h10 = cal_performance(ranking)
            

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
            print(f"\nCovering head ratio in val set: {cnt}/{total} = {cnt/total}\n")
            v_head_ratio = cnt/total

        else:
            v_mrr, v_h1, v_h2, v_h3, v_h10 = -1, -1, -1, -1, -1
        
        # eval on test set
        if eval_test:
            print("evaluating on test set...")
            
        else:
            t_mrr, t_h1, t_h2, t_h3, t_h10 = -1, -1, -1, -1, -1
        # print("goodbye")
        i_time = time.time() - i_time
        out_str = '[TRAIN] MRR:%.4f H@1:%.4f H@2:%.4f H@3:%.4f H@10:%.4f\t [VALID] MRR:%.4f H@1:%.4f H@2:%.4f H@3:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@2:%.4f H@3:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(tr_mrr, tr_h1, tr_h2, tr_h3, tr_h10, v_mrr, v_h1, v_h2, v_h3, v_h10, t_mrr, t_h1, t_h2, t_h3, t_h10, self.t_time, i_time)
        return v_mrr, {'tr_mrr': tr_mrr, 'tr_h1': tr_h1, 'tr_h2': tr_h2, 'tr_h3': tr_h3, 'tr_h10': tr_h10, 'v_mrr': v_mrr, 'v_h1': v_h1, 'v_h2': v_h2, 'v_h3': v_h3, 'v_h10': v_h10, 'out_str': out_str, 'tr_head_ratio': tr_head_ratio, 'v_head_ratio': v_head_ratio}

def train(trainer, args, resume_from=None):
    """Train the model with optional resume capability.
    
    Args:
        trainer: Trainer object
        args: Training arguments
        resume_from: Path to checkpoint file to resume training from. If None, start fresh.
    """
    # WANDB: Init
    wandb.init(
        project=args.project_name,
        config={
            "gnn_lr": args.gnn_lr,
            "projector_lr": args.projector_lr,
            "epochs": args.epochs,
            "gnn_layers": GNN_config.n_layer,
            "proj_layers": Projector_config.n_layers
        }
    )

    best_mrr, bearing = 0, 0
    best_str = ""
    start_epoch = 0
    
    # Load checkpoint if resuming
    if resume_from is not None:
        checkpoint_state = trainer.load_checkpoint(resume_from)
        start_epoch = checkpoint_state['epoch'] + 1
        best_mrr = checkpoint_state['best_mrr']
        print(f"Resuming training from epoch {start_epoch} with best_mrr={best_mrr}")

    for epoch in range(start_epoch, args.epochs):
        mrr, metrics, out_str = trainer.train_batch(epoch)
            
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(f"Epoch {epoch} | New Best Var MRR: {metrics['v_mrr']:.4f} | Train MRR: {metrics['tr_mrr']:.4f}")
            bearing = 0
            
            # Save checkpoint
            best_tag = f'ValMRR_{str(mrr)[:5]}'
            trainer.saveModelToFiles(args, best_tag, epoch, best_mrr)
            
            # WANDB: Track best metric
            wandb.run.summary["best_mrr"] = best_mrr
        else:
            bearing += 1
            
        if bearing >= args.bearing: 
            print(f'Early stopping at epoch {epoch+1}.')
            break
    
    print("Training finished.")
    print(f"Best Results: {best_str}")
    wandb.finish()
    return best_mrr

class GNN_config:
    n_layer = 8
    active_layer = 3
    hidden_dim = 64
    attn_dim = 4
    dropout = 0.3
    n_ent = 1024 # subgraph size
    n_rel = 237
    shortcut = True
    readout = 'linear'
    concatHidden = True
    initializer = 'relation'
    llm_description_emb_path = "data_for_CL/llm_description_aligned_emb.pkl"
    ## 474 relations / 20 versions each / 4096 dimensional
    with open(llm_description_emb_path, "rb") as f:
        llm_description_aligned_emb = pkl.load(f)
    llm_emb = list(llm_description_aligned_emb.values())
    llm_emb = torch.stack(llm_emb, dim=0)
    # pretrain_model_path = "topk_0.1_layer_8_ValMRR_0.437.pt"
    del(llm_description_aligned_emb)

class Projector_config:
    n_layers = 8
    in_dim = 4096
    hidden_dims = [512, 256]
    out_dim = 64
    # pretrain_model_path = "weights/pretrain/projector/best_projector_HNM.pt"

class Training_args:
    project_name = "train_align_finetune"
    gnn_lr = 5e-4
    projector_lr = 5e-4
    temperature = 0.07
    epochs = 50
    bearing = 8
    device = 'cuda:0'



if __name__ == "__main__":

    # os.environ["WANDB_MODE"] = "disabled"

    with open("data_for_GNN_finetune_another_way.pkl", "rb") as f:
        data = pkl.load(f)
    data = data[:int(0.1 * len(data))]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    seed = 21
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Worker init function for DataLoader
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_len = int(len(data) * 0.8)
    train_data = data[:train_len]
    val_data = data[train_len:]
    
    
    # train_rel = []
    # for data in train_data:
    #     train_rel.extend([a[1] for a in data['drop_edges']])
    # train_rel = torch.unique(torch.tensor(train_rel).flatten()).tolist()
    # val_rel = []
    # for data in val_data:
    #     val_rel.extend([a[1] for a in data['drop_edges']])
    # val_rel = torch.unique(torch.tensor(val_rel).flatten()).tolist()
    # val_rel = set(val_rel) - set(train_rel)
    # print("Number of new relations in val set:", len(val_rel))
    # exit()

    # Create generator for DataLoader reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader2(train_data, mode='train')
    val_train_loader = DataLoader2(train_data, mode='val')
    train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=True, collate_fn=train_loader.collate_fn, generator=g)   

    val_loader = DataLoader2(val_data, mode='val')  
    val_dataloader = DataLoader(val_loader, batch_size=32, shuffle=False, collate_fn=val_loader.collate_fn, )
    
    val_train_loader = DataLoader(val_train_loader, batch_size=32, shuffle=False, collate_fn=val_train_loader.collate_fn)
    gnn_model = GNN_auto(GNN_config, train_loader)
    if "pretrain_model_path" in GNN_config.__dict__:
        gnn_model = loadModel(GNN_config.pretrain_model_path, gnn_model)
        print("GNN Model loaded successfully.")
    projector_model = Projector(Projector_config.in_dim, Projector_config.hidden_dims, Projector_config.out_dim, Projector_config.n_layers)
    if "pretrain_model_path" in Projector_config.__dict__:
        projector_model = loadModel(Projector_config.pretrain_model_path, projector_model)
        print("Projector Model loaded successfully.")
    # print("GNN Model Structure:")
    trainer = Trainer(gnn_model, projector_model, train_dataloader, val_dataloader, val_train_loader, Training_args)
    
    # Resume from checkpoint: Uncomment to resume from the latest checkpoint
    # checkpoint_path = get_latest_checkpoint()
    # if checkpoint_path:
    #     print(f"Found checkpoint: {checkpoint_path}")
    #     train(trainer, Training_args, resume_from=checkpoint_path)
    # else:
    #     print("No checkpoint found, starting fresh training.")
    #     train(trainer, Training_args)
    
    # # Or specify a specific checkpoint path:
    # train(trainer, Training_args, resume_from='saveModels/topk_ValMRR_0.123.pt')
    
    # Start fresh training (default)
    train(trainer, Training_args)
        
