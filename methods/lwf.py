import torch
import torch.nn.functional as F
# from focal_loss.focal_loss import FocalLoss
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, eval_node_classifier

# Utilities
from progressbar import progressbar
from methods.utility import get_graph_class_ratio
from backbones.gcn import GCN

import copy

class LwF():
    def __init__(self, model, tasks, device, focal):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.focal = focal

    def train(self, epoch, tim=True):
        model = self.model
        tasks = self.tasks
        device = self.device
        opt = self.opt
        
        previous_model = copy.deepcopy(model)
        previous_model.initialize()

        criterion = torch.nn.CrossEntropyLoss()
        # criterion = FocalLoss(gamma=self.focal)
        T = 5.0
        alpha = 0.4

        performace_matrix = torch.zeros(len(tasks), len(tasks))
        for k in progressbar(range(len(tasks), redirect_stdout=True)):
            task = tasks[k]
            last_cls = torch.unique(task.y)[-1]
            incremental_cls = last_cls + 1

            task.to(device, "x", "y", "adj_t")
            model.train()
            self.opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
            for epoch in range(epoch):
                outputs = model(task)[:, :incremental_cls]
                soft_target = previous_model(task)[:, :incremental_cls]
                targets = task.y[task.train_mask]

                # Cross entropy between output of the new task and label
                loss1 = criterion(outputs[task.train_mask], targets)
                # Using the new softmax to handle outputs
                outputs_S = F.softmax(outputs[task.train_mask] / T, dim=1)[:, :(incremental_cls - 2)]
                outputs_T = F.softmax(soft_target[task.train_mask] / T, dim=1)[:, :(incremental_cls - 2)]
                # Cross entropy between output of the old task and output of the old model
                loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
                loss2 = loss2.sum(1)
                loss2 = loss2.mean() * T * T
                if k == 0:
                    loss = loss1
                else:
                    # loss = loss2
                    loss = loss1 * alpha + loss2 * (1 - alpha)
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
            previous_model = copy.deepcopy(model)

            # Save the GPU memory for the evaluation phase.
            task.cpu() 
            task.cpu()

            # Test the model from task 0 to task k
            accs = []
            for k_ in range(k + 1):
                task_ = tasks[k_].to(device, "x", "y", "adj_t")
                acc = eval_node_classifier(model, task_) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}",end="|")
                performace_matrix[k, k_] = acc
            print(f"AP: {sum(accs) / len(accs):.2f}")
        return self.memory_bank, performace_matrix
    
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