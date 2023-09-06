import torch
import torch.nn.functional as F
# from focal_loss.focal_loss import FocalLoss


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            # torch.nn.init.normal_(layer.lin.weight.data)
            layer.reset_parameters()
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x
        # return F.log_softmax(x, dim=1)

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
        return x
    
    def encode_noise(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_t)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        random_noise = torch.rand_like(x).cuda()
        x += torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x

def train_node_classifier(model, data, optimizer, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        if incremental_cls:
            out = model(data)[:, incremental_cls[0]:incremental_cls[1]]
        else:
            out = model(data) 
        
        loss = ce(out[data.train_mask], data.y[data.train_mask]-incremental_cls[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    pred = model(data)[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
    correct = (pred == data.y[data.test_mask]-incremental_cls[0]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
