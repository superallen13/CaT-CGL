import torch
from backbones.gnn import GNN
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv


class GCN(GNN):
    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__()
        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid))  # hidden layers
            self.layers.append(GCNConv(nhid, nout))  # output layers

def main():
    from backbones.gnn import train_node_classifier, eval_node_classifier
    from torch_geometric.datasets import CoraFull, Reddit
    dataset = Reddit("/home/s4669928/graph/cat/data")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]
    nin = data.x.shape[1]
    nhid = 256
    nout = data.y.shape[0]
    nlayers = 2
    model = GCN(nin, nhid, nout, nlayers).to(device)
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