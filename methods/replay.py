import torch
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, eval_node_classifier

# Utilities
from progressbar import progressbar
from methods.utility import get_graph_class_ratio
from backbones.gcn import GCN

class Replay():
    def __init__(self, model, tasks, budget, m_update, device, focal, pseudo_label, retrain):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.budgets = self._assign_buget_per_cls(budget)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.memory_bank = []
        self.strategy = m_update
        self.focal = focal
        self.pseudo_label = pseudo_label
        self.retrain = retrain

    def train(self, epoch, tim=True):
        model = self.model
        tasks = self.tasks
        device = self.device
        opt = self.opt

        performace_matrix = torch.zeros(len(tasks), len(tasks))
        for k in progressbar(range(len(tasks)), redirect_stdout=True):
            task = tasks[k]
            last_cls = torch.unique(task.y)[-1]

            if self.pseudo_label:
                task = self._add_pseudo_labels(task)

            # Get the replayed graph for the current task.
            replayed_graph = self.memorize(task, self.budgets[k])
            self.memory_bank.append(replayed_graph)  # Update the memory bank.

            # task.cpu()

            # # Batch graphs for training.
            # if tim:
            #     if self.strategy == "all":
            #         replayed_graphs = Batch.from_data_list(self.memory_bank)
            #     elif self.strategy == "onlyCurrent":
            #         replayed_graphs = Batch.from_data_list([self.memory_bank[-1]])
            # else:
            #     if k == 0:
            #         replayed_graphs = Batch.from_data_list([task])
            #     else:
            #         replayed_graphs = Batch.from_data_list(self.memory_bank[:-1] + [task])
            # replayed_graphs.to(device, "x", "y", "adj_t")

            # if self.retrain:
            #     model.initialize()

            # # train
            # model = train_node_classifier(model, 
            #                               replayed_graphs, 
            #                               opt, 
            #                               n_epoch=epoch, 
            #                               incremental_cls=last_cls+1, focal=self.focal)
            
            # Save the GPU memory for the evaluation phase.
            # replayed_graphs.cpu() 

            # Test the model from task 0 to task k
            # accs = []
            # for k_ in range(k + 1):
            #     task_ = tasks[k_].to(device, "x", "y", "adj_t")
            #     acc = eval_node_classifier(model, task_) * 100
            #     accs.append(acc)
            #     task_.to("cpu")
            #     if self.strategy == "all":
            #         print(f"T{k_} {acc:.2f}",end="|")
            #     elif self.strategy == "onlyCurrent":
            #         if k == k_:
            #             print(f"T{k_} {acc:.2f}",end="|")
            #     performace_matrix[k, k_] = acc
            # print(f"AP: {sum(accs) / len(accs):.2f}")
        return self.memory_bank, performace_matrix
    
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