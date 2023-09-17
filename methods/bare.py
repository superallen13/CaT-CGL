import torch
import numpy as np
from backbones.gnn import train_node_classifier, eval_node_classifier

class Bare():
    def __init__(self, model, tasks, device):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k]  # Load the incoming graph.

            task.to(self.device, "x", "y", "adj_t")
            num_cls = torch.unique(task.y)[-1]
            model = train_node_classifier(self.model, task, self.opt, weight=None, n_epoch=epoches, incremental_cls=(0, num_cls+1))

            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(self.device, "x", "y", "adj_t")
                if IL == "classIL":
                    acc = eval_node_classifier(self.model, task_, incremental_cls=(0, num_cls+1)) * 100
                else:
                    num_cls = torch.unique(task_.y)[-1]
                    acc = eval_node_classifier(self.model, task_, incremental_cls=(num_cls+1-2, num_cls+1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=True)
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            APs.append(AP)
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, np.mean(APs), AF
