import torch
import numpy as np
from backbones.gnn import eval_node_classifier


class MAS():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # for MAS
        self.reg = args['memory_strength']
        self.fisher = []
        self.optpar = []
        self.ce = torch.nn.CrossEntropyLoss()
        self.n_seen_examples = 0

    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            num_cls = torch.unique(task.y)[-1] + 1
            n_new_examples = task.x[task.train_mask].shape[0]
            for _ in range(epoches):
                n_per_cls = [(task.y[task.train_mask] == cls).sum() for cls in range(num_cls)]
                loss_w = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                output = self.model(task)[task.train_mask, :num_cls]
                ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_w).to(self.device))
                loss = ce(output, task.y[task.train_mask])

                if k > 0:
                    for i, p in enumerate(self.model.parameters()):
                        l = self.reg * self.fisher[i]
                        l = l * (p - self.optpar[i]).pow(2)
                        loss += l.sum()

                self.model.zero_grad()
                loss.backward()
                self.opt.step()
            
            self.optpar = []
            new_fisher = []
            output = self.model(task)[task.train_mask, :num_cls]

            output.pow_(2)
            loss = output.mean()
            self.model.zero_grad()
            loss.backward()

            for p in self.model.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                new_fisher.append(pg)
                self.optpar.append(pd)

            if len(self.fisher) != 0:
                for i, f in enumerate(new_fisher):
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i] * n_new_examples) / (self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                self.fisher = new_fisher
                self.n_seen_examples = n_new_examples
            
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
