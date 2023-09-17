import torch
import numpy as np
from backbones.gat import eval_node_classifier


class TWP():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.ce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}
        self.mem_mask = None
        # hyper-parameters
        self.lambda_l = args['lambda_l']
        self.lambda_t = args['lambda_t']
        self.beta = args['beta']


    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)
            num_cls = torch.unique(task.y)[-1]
            self.model.train()
            for _ in range(epoches):
                output, _ = self.model(task)
                loss = self.ce(output[task.train_mask, :num_cls+1], task.y[task.train_mask])
                self.model.zero_grad()
                loss.backward(retain_graph=True)

                grad_norm = 0
                for p in self.model.parameters():
                    pg = p.grad.data.clone()
                    grad_norm += torch.norm(pg, p=1)

                for k_ in range(k):
                    for i, p in enumerate(self.model.parameters()):
                        l = self.lambda_l * self.fisher_loss[k_][i] + self.lambda_t * self.fisher_att[k_][i]
                        l = l * (p - self.optpar[k_][i]).pow(2)
                        loss += l.sum()

                loss += self.beta * grad_norm
                loss.backward()
                self.opt.step()
            
            # After the last epoch.
            self.fisher_loss[k] = []
            self.fisher_att[k] = []
            self.optpar[k] = []

            output, att_w = self.model(task)
            loss = self.ce(output[task.train_mask, :num_cls+1], task.y[task.train_mask])
            self.model.zero_grad()
            loss.backward(retain_graph=True)

            for p in self.model.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[k].append(pd)
                self.fisher_loss[k].append(pg)

            eloss = torch.norm(att_w.storage._value)
            eloss.backward()
            for p in self.model.parameters():
                pg = p.grad.data.clone().pow(2)
                self.fisher_att[k].append(pg)
            
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
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            APs.append(AP)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, np.mean(APs), AF
