from methods.replay import Replay
import torch
import random
from backbones.gcn import GCN
# from progressbar import progressbar
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import copy

class RandomSubgraphSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, center_node_budget, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, center_node_budget)
        return center_nodes_selected

    def node_sampler(self, ids_per_cls_train, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        budget_ = min(budget, int(max_ratio_per_cls * len(ids_per_cls_train)))
        store_ids.extend(random.sample(ids_per_cls_train, budget_))
        return store_ids


class degree_based_sampler(torch.nn.Module):
    # based on random walk sasmpler01, sample a subgraph based on the degrees of the neighbors
    def __init__(self, args):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, gnn, ids_per_cls, restart=0.0):
        center_nodes_selected = self.node_sampler(ids_per_cls, graph, center_node_budget)
        all_nodes_selected = self.nei_sampler(center_nodes_selected, graph, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self,ids_per_cls_train, graph, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        for i, ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(
                budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

    def nei_sampler(self, center_nodes_selected, graph, nei_budget):
        probs = graph.in_degrees().float()
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)
        for b in nei_budget:
            if b==0:
                continue
            # from 1-hop to len(nei_budget)-hop neighbors
            neighbors = list(set(graph.in_edges(nodes_selected_current_hop)[0].tolist()))
            # remove selected nodes
            for n in retained_nodes:
                neighbors.remove(n)
            if len(neighbors)==0:
                continue
            prob = probs[neighbors]
            sampled_neibs_ = torch.multinomial(prob, min(b, len(neighbors)), replacement=False).tolist()
            sampled_neibs = torch.tensor(neighbors)[sampled_neibs_] # map the ids to the original ones
            retained_nodes.extend(sampled_neibs.tolist())
        return list(set(retained_nodes))

class SSM(Replay):
    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__(model, tasks, budget, m_update, device)

    def memorize(self, task, budgets):
        classes = torch.unique(task.y)
        ids_per_cls_train = []
        for cls in classes:
            cls_train_mask = (task.y == cls).logical_and(task.train_mask)
            ids_per_cls_train.append(cls_train_mask.nonzero(as_tuple=True)[0].tolist())

        sampler = RandomSubgraphSampler()
        nodes_sampled = []
        for i, ids in enumerate(ids_per_cls_train):
            c_nodes_sampled = sampler(budgets[i], ids)
            nodes_sampled += c_nodes_sampled

        replayed_graph = task.subgraph(torch.tensor(nodes_sampled))
        edge_index = replayed_graph.edge_index
        adj = SparseTensor(row=edge_index[0], 
                           col=edge_index[1], 
                           sparse_sizes=(replayed_graph.num_nodes, replayed_graph.num_nodes))
        replayed_graph.adj_t = adj.t()
        return replayed_graph
        
