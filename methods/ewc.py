import torch
import numpy as np
from backbones.gnn import eval_node_classifier

class EWC():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # for EWC
        self.reg = args['memory_strength']
        self.fisher = {}
        self.optpar = {}
        self.ce = torch.nn.CrossEntropyLoss()

    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            num_cls = torch.unique(task.y)[-1]

            # Train
            for _ in range(epoches):
                output = self.model(task)[task.train_mask, :num_cls+1]
                loss = self.ce(output, task.y[task.train_mask])  # main loss
                # regularization item
                for k_ in range(k):
                    for i, p in enumerate(self.model.parameters()):
                        l = self.reg * self.fisher[k_][i]
                        l = l * (p - self.optpar[k_][i]).pow(2)
                        loss += l.sum()

                self.model.zero_grad()
                loss.backward()
                self.opt.step()

            
            output = self.model(task)[task.train_mask, :num_cls+1]
            self.model.zero_grad()
            self.ce(output, task.y[task.train_mask]).backward()

            self.fisher[k] = []
            self.optpar[k] = []
            for p in self.model.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.fisher[k].append(pg)
                self.optpar[k].append(pd)
            
            # Evaluation
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
