import torch
import random

from torch_geometric.transforms import RandomNodeSplit
from torch_sparse import SparseTensor
from progressbar import progressbar



class Streaming():
    def __init__(self, cls_per_task, dataset):
        self.cls_per_task = cls_per_task
        self.tasks = self.prepare_tasks(dataset)
        self.n_tasks = len(self.tasks)
    
    def prepare_tasks(self, dataset):
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

            tasks.append(subgraph)
        return tasks
