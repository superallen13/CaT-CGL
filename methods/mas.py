import torch
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, eval_node_classifier

# Utilities
from progressbar import progressbar
from methods.utility import get_graph_class_ratio
from backbones.gcn import GCN

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

    def observer(self, epoches):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            max_cls = torch.unique(task.y)[-1]
            n_new_examples = task.x[task.train_mask].shape[0]
            for _ in range(epoches):
                output = self.model(task)[task.train_mask, :max_cls+1]
                loss = self.ce(output, task.y[task.train_mask])

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
            output = self.model(task)[task.train_mask, :max_cls+1]

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
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i]*n_new_examples) / (self.n_seen_examples + n_new_examples)
                self.n_seen_examples += n_new_examples
            else:
                for i, f in enumerate(new_fisher):
                    self.fisher.append(new_fisher[i])
                self.n_seen_examples = n_new_examples
            
            accs = []
            AF = 0
            for k_ in range(k + 1):
                task_ = tasks[k_].to(self.device, "x", "y", "adj_t")
                max_cls = torch.unique(task_.y)[-1]
                acc = eval_node_classifier(self.model, task_, incremental_cls=(0, max_cls+1)) * 100
                accs.append(acc)
                task_.to("cpu")
                print(f"T{k_} {acc:.2f}", end="|", flush=True)
                performace_matrix[k, k_] = acc
            AP = sum(accs) / len(accs)
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, AF
