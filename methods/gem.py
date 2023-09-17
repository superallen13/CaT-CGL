import torch
import quadprog
import numpy as np
from torch_sparse import SparseTensor
from backbones.gnn import eval_node_classifier

def store_grad(para, grads, grad_dims, task_id):
    grads[:, task_id].fill_(0.0)
    cnt = 0
    for p in para():
        if p.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, task_id].copy_(p.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.detach().cpu().t().double().numpy()
    gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class GEM():
    def __init__(self, model, tasks, device, args):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.device = device
        self.ce = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # for GEM
        self.margin = args['memory_strength']
        self.memory_budget = int(args['n_memories'])
        self.memories = []

        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), len(tasks)).to(device)

    def observer(self, epoches, IL):
        tasks = self.tasks
        performace_matrix = torch.zeros(len(tasks)+1, len(tasks)+1)
        APs = []
        for k in range(len(tasks)):
            task = tasks[k]  # Load the incoming graph.

            # Update the memory bank.
            train_ids = task.train_mask.nonzero(as_tuple=True)[0].tolist()
            sampled_ids = np.random.choice(train_ids, self.memory_budget, replace=False)
            replayed_graph = task.subgraph(torch.tensor(sampled_ids))
            edge_index = replayed_graph.edge_index
            adj = SparseTensor(row=edge_index[0], 
                               col=edge_index[1], 
                               sparse_sizes=(replayed_graph.num_nodes, replayed_graph.num_nodes))
            replayed_graph.adj_t = adj.t()
            self.memories.append(replayed_graph)

            for _ in range(epoches):
                # Calculate gradients for memories.
                for i, memory in enumerate(self.memories):
                    memory = memory.to(self.device, "x", "y", "adj_t")
                    num_cls = torch.unique(memory.y)[-1] + 1
                    output = self.model(memory)[memory.train_mask, :num_cls]
                    old_task_loss = self.ce(output, memory.y[memory.train_mask])
                    self.model.zero_grad()
                    old_task_loss.backward()
                    store_grad(self.model.parameters, self.grads, self.grad_dims, i)

                # Calucate the gradients for the current task.
                num_cls = torch.unique(task.y)[-1] + 1
                task = task.to(self.device, "x", "y", "adj_t")
                output = self.model(task)[task.train_mask, :num_cls]
                loss = self.ce(output, task.y[task.train_mask])
                self.model.zero_grad()
                loss.backward()

                # check if gradient violates constraints
                if k > 0:
                    # copy gradient
                    store_grad(self.model.parameters, self.grads, self.grad_dims, k)
                    indx = torch.cuda.LongTensor(list(range(k)))
                    dotp = torch.mm(self.grads[:, k].unsqueeze(0),
                                    self.grads.index_select(1, indx))
                    if (dotp < 0).sum() != 0:
                        project2cone2(self.grads[:, k].unsqueeze(1), 
                                      self.grads.index_select(1, indx), 
                                      self.margin)
                        # copy gradients back
                        overwrite_grad(self.model.parameters, self.grads[:, k], self.grad_dims)
                self.opt.step()
            
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
            APs.append(APs)
            for t in range(k):
                AF += performace_matrix[k, t] - performace_matrix[t, t]
            AF = AF / k if k != 0 else AF
            print(f"AF: {AF:.2f}", flush=True)
        return AP, np.mean(APs), AF
