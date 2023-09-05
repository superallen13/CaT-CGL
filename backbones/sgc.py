import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_sparse import SparseTensor
from backbones.gcn import GCN


class SGC(GCN):
    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__(nin, nhid, nout, nlayers)


    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers:
            x = layer(x, adj_t)
        return x
        # return F.log_softmax(x, dim=1)

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
        return x
