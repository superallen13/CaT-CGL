import torch
from methods.replay import Replay
from backbones.gcn import GCN
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class ERGNN(Replay):
    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__(model, tasks, budget, m_update, device)

    def memorize(self, task, budgets):
        classes = torch.unique(task.y)
        ids_per_cls_train = []
        for cls in classes:
            cls_train_mask = (task.y == cls).logical_and(task.train_mask)
            ids_per_cls_train.append(cls_train_mask.nonzero(as_tuple=True)[0].tolist())

        # init a encoder
        encoder = GCN(task.num_features, 512, 256, 2).to(self.device)
        emb = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device))

        centers = [emb[ids].mean(0) for ids in ids_per_cls_train]
        sim = [centers[i].view(1, -1).mm(emb[ids_per_cls_train[i]].permute(1, 0)).squeeze() for i in range(len(centers))]
        rank = [s.sort()[1].tolist() for s in sim]
        ids_selected = []
        for i, ids in enumerate(ids_per_cls_train):
            nearest = rank[i][0:min(budgets[i], len(ids_per_cls_train[i]))]
            ids_selected.extend([ids[i] for i in nearest])
        
        replayed_graph_size = len(ids_selected)
        replayed_graph = Data(x=task.x[ids_selected], y=task.y[ids_selected], adj_t=SparseTensor.eye(replayed_graph_size, replayed_graph_size).t())
        replayed_graph.train_mask = torch.ones(replayed_graph_size, dtype=torch.bool)
        return replayed_graph
