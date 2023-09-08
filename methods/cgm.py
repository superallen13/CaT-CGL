# DL and GL
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor

# Own modules
from methods.replay import Replay
from backbones.gcn import GCN
from backbones.encoder import Encoder
from methods.utility import get_graph_class_ratio
from backbones.gnn import train_node_classifier

# Utilities
from torch_geometric.loader import NeighborLoader
from .utility import *
import random
# from progressbar import progressbar


class CGM(Replay):
    def __init__(self, model, tasks, budget, m_update, device, args):
        super().__init__(model, tasks, budget, m_update, device)
        self.n_encoders = args['n_encoders']
        self.feat_lr = args['feat_lr']
        self.hid_dim = args['hid_dim']
        self.emb_dim = args['emb_dim']
        self.n_layers = args['n_layers']
        self.feat_init = "randomChoice"
        self.feat_init = args["feat_init"]
        self.hop = args['hop']
        
    def memorize(self, task, budgets):
        labels_cond = []
        for i, cls in enumerate(task.classes):
            # cls_train_num = task.y[task.train_mask][task.y[task.train_mask]==cls].shape[0]
            # print(cls_train_num)
            # if cls_train_num < budgets[i]:
            #     budgets[i] = cls_train_num
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)

        feat_cond = torch.nn.Parameter(torch.FloatTensor(sum(budgets), task.num_features))
        feat_cond = self._initialize_feature(task, budgets, feat_cond, self.feat_init)

        replayed_graph = self._condense(task, feat_cond, labels_cond, budgets)
        return replayed_graph
    
    def _initialize_feature(self, task, budgets, feat_cond, method="randomChoice"):
        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)
        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += random.choices(ids_at_cls, k=budgets[i])
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        elif method == "kMeans":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += query(task, ids_at_cls, budgets[i], self.device)
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        return feat_cond

    def _condense(self, task, feat_cond, labels_cond, budgets):
        self_loops = SparseTensor.eye(sum(budgets), sum(budgets)).t()
        opt_feat = torch.optim.Adam([feat_cond], lr=self.feat_lr)

        cls_train_masks = []
        for cls in task.classes:
            cls_train_masks.append((task.y == cls).logical_and(task.train_mask))   
        
        encoder = Encoder(task.num_features, self.hid_dim, self.emb_dim, self.n_layers, self.hop).to(self.device)
        for _ in range(self.n_encoders):
            encoder.initialize()
            with torch.no_grad():
                emb_real = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device))
                emb_real = F.normalize(emb_real)
            emb_cond = encoder.encode_without_e(feat_cond.to(self.device))
            emb_cond = F.normalize(emb_cond)

            loss = torch.tensor(0.).to(self.device)
            for i, cls in enumerate(task.classes):
                real_emb_at_class = emb_real[cls_train_masks[i]]
                cond_emb_at_class = emb_cond[labels_cond == cls]
                
                dist = torch.mean(real_emb_at_class, 0) - torch.mean(cond_emb_at_class, 0)
                loss += torch.sum(dist ** 2)
        
            # Update the feature matrix
            opt_feat.zero_grad()
            loss.backward()
            opt_feat.step()

        # Wrap the graph data object
        replayed_graph = Data(x=feat_cond.detach().cpu(), 
                              y=labels_cond, 
                              adj_t=self_loops)
        replayed_graph.train_mask = torch.ones(sum(budgets), dtype=torch.bool)
        return replayed_graph