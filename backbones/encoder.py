from backbones.gnn import GNN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.typing import torch_sparse


def gcn_norm(adj_t):
    adj_t = torch_sparse.fill_diag(adj_t, 1.)  # add self-loops.
    
    # Normalization.
    deg = torch_sparse.sum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

    return adj_t


class Linear(GCNConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__(in_channels, out_channels)
    
    def forward(self, x):
        return self.lin(x)
    

class Encoder(GNN):
    def __init__(self, nin, nhid, nout, nlayers, hop, activation=True):
        super().__init__()
        self.feat_agg = None
        self.hop = hop
        self.activation = activation  # True or False
        if nlayers == 1:
            self.layers.append(Linear(nin, nout))
        else:
            self.layers.append(Linear(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(Linear(nhid, nhid))  # hidden layers
            self.layers.append(Linear(nhid, nout))  # output layers

    def encode_without_e(self, x):
        self.eval()
        for layer in self.layers[:-1]:  # without the FC layer.
            x = layer(x)
            if self.activation:
                x = F.relu(x)
        return x

    def encode(self, x, adj_t):
        if self.feat_agg is None:
            adj_t = gcn_norm(adj_t)
            for _ in range(self.hop):
                x = self.layers[0].propagate(adj_t, x=x)
            self.feat_agg = x

        self.eval()
        x = self.feat_agg
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x)
            if self.activation:
                x = F.relu(x)
        return x