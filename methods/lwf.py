import torch
import numpy as np
from backbones.gnn import eval_node_classifier

# Utilities
import copy
from torch.autograd import Variable


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

class LWF():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        self.T = args['T']
        self.lambda_dist = args['lambda_dist']

    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k].to(self.device)

            # Train
            for _ in range(epoches):
                self.model.train()
                num_cls = torch.unique(task.y)[-1] + 1
                output = self.model(task)
                loss = self.ce(output[task.train_mask, :num_cls], task.y[task.train_mask])
                
                if k > 0:
                    num_cls_ = torch.unique(tasks[k-1].y)[-1] + 1
                    target_ = prev_model(task)[task.train_mask, :num_cls_]
                    dist_loss = MultiClassCrossEntropy(output[task.train_mask, :num_cls_], target_, self.T)
                    loss += self.lambda_dist * dist_loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            
            prev_model = copy.deepcopy(self.model)
            
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
            print(f"AP: {AP:.2f}", end=", ", flush=True)
            APs.append(AP)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, np.mean(APs), AF
