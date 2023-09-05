from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import progressbar
import torch
import random


class Streaming():
    def __init__(self, task, cls_per_task, dataset, unlabeled_rate):
        self.cls_per_task = cls_per_task
        self.unlabeled_rate = unlabeled_rate
        self.tasks = self.prepare_tasks(task, dataset)
        self.n_tasks = len(self.tasks)
    
    def prepare_tasks(self, task, dataset):
        graph = dataset[0]
        tasks = []
        n_tasks = int(dataset.num_classes / self.cls_per_task)
        for k in progressbar(range(n_tasks), redirect_stdout=True):        
            start_cls = k * self.cls_per_task
            classes = list(range(start_cls, start_cls + self.cls_per_task))
            subset = sum(graph.y == cls for cls in classes).squeeze().nonzero(as_tuple=False)
            subgraph = graph.subgraph(subset)
            
            # Split to train/val/test
            transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
            subgraph = transform(subgraph)

            edge_index = subgraph.edge_index
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(subgraph.num_nodes, subgraph.num_nodes))
            subgraph.adj_t = adj.t().to_symmetric()  # Arxiv is an directed graph.

            subgraph.task_id = k
            subgraph.classes = classes

            # Modify the unlabeled node.
            if self.unlabeled_rate > 0:
                train_mask = torch.zeros(subgraph.train_mask.shape[0], dtype=torch.bool)
                train_ids = torch.nonzero(subgraph.train_mask).squeeze().tolist()
                for cls in classes:
                    cls_ids = torch.nonzero(subgraph.y == cls).squeeze().tolist()
                    train_size = len(cls_ids)
                    labeled_num = int(train_size * (1 - self.unlabeled_rate))
                    if labeled_num == 0:
                        labeled_num = 1
                    unlabeled_ids = random.sample(cls_ids, k=labeled_num)
                    train_mask[unlabeled_ids] = True
                subgraph.train_mask = train_mask

            tasks.append(subgraph)
        return tasks
