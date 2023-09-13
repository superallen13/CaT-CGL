import torch
from backbones.gnn import GNN
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(GNN):
    def __init__(self, nin, nhid, nout, nlayers, concat=False, heads=8):
        super().__init__()
        if nlayers == 1:
            self.layers.append(GATConv(nin, nout, concat=concat, heads=1))
        else:
            self.layers.append(GATConv(nin, nhid, concat=concat, heads=heads))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(GATConv(nhid, nhid, concat=concat, heads=heads))  # hidden layers
            self.layers.append(GATConv(nhid, nout, concat=concat, heads=1))  # output layers
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x, att_w = self.layers[-1](x, adj_t, return_attention_weights=True)
        return x, att_w

def train_node_classifier(model, data, optimizer, weight=None, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for epoch in range(n_epoch):
        out, _ = model(data)
        if incremental_cls:
            out = out[:, 0:incremental_cls[1]]
        loss = ce(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    out, _ = model(data)
    if incremental_cls:
        pred = out[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
        correct = (pred == data.y[data.test_mask]-incremental_cls[0]).sum()
    else:
        pred = out[data.test_mask].argmax(dim=1)
        correct = (pred == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

def main():
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="/home/s4669928/graph/CaT-CGL/data", name="CiteSeer")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]
    nin = data.x.shape[1]
    nhid = 32
    nout = data.y.shape[0]
    nlayers = 2
    model = GAT(nin, nhid, nout, nlayers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    edge_index = data.edge_index
    adj = SparseTensor(row=edge_index[0], 
                       col=edge_index[1], 
                       sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = adj.t()
    data.to(device, "x", "y", "adj_t")
    model = train_node_classifier(model, data, optimizer)
    acc = eval_node_classifier(model, data)
    print(f"acc is {acc:0.2f}")

if __name__ == '__main__':
    main()