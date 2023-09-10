import torch
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, eval_node_classifier

# Utilities
from progressbar import progressbar
from methods.utility import get_graph_class_ratio
from backbones.gcn import GCN

class Replay():
    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.budgets = self._assign_buget_per_cls(budget)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.memory_bank = []
        self.strategy = m_update

    def observer(self):
        tasks = self.tasks

        # performace_matrix = torch.zeros(len(tasks), len(tasks))
        for k in progressbar(range(len(tasks)), redirect_stdout=True):
        # for k in range(len(tasks)):
            task = tasks[k]
            # Get the replayed graph for the current task.
            replayed_graph = self.memorize(task, self.budgets[k])
            self.memory_bank.append(replayed_graph)  # Update the memory bank.

        return self.memory_bank
    
    def memorize(self, task, budgets):
        raise NotImplementedError("Please implement this method!")
    
    def _assign_buget_per_cls(self, budget):
        budgets = []
        for task in self.tasks:
            if budget is None:
                budgets.append([])
            else:
                classes = torch.unique(task.y)
                budgets_at_task = []
                for cls in classes:
                    class_ratio = get_graph_class_ratio(task, cls)
                    replay_cls_size = int(budget * class_ratio)
                    if replay_cls_size == 0:
                        budgets_at_task.append(1)
                    else:
                        budgets_at_task.append(replay_cls_size)
                # Because using the int(), sometimes, there are still some nodes which are not be allocated a label.
                gap = budget - sum(budgets_at_task)

                for i in range(gap):
                    budgets_at_task[i % len(classes)] += 1
                budgets.append(budgets_at_task)
        return budgets
    
    def _add_pseudo_labels(self, task):
        max_cls = int(torch.max(torch.unique(task.y)).item())
        model = GCN(task.x.shape[1], 512, max_cls+1, 2).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model = train_node_classifier(model, 
                                      task.to(self.device), 
                                      opt, 
                                      n_epoch=500)
        model.eval()
        pred = model(task).argmax(dim=1)
        pred[task.train_mask] = task.y[task.train_mask]
        task.pseudo_labels = pred  # Modify the labeled data.
        return task.to('cpu')