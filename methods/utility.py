def get_graph_class_ratio(graph, class_id, pseudo=False):
    if pseudo:
        labels = graph.pseudo_labels
    else:
        labels = graph.y[graph.train_mask]
    return (labels == class_id).sum().item() / len(labels)

import torch
# from fast_pytorch_kmeans import KMeans
from backbones.gcn import GCN

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def query(task, idx, n, device):
    encoder = GCN(task.num_features, 256, 128, 2).to(device)
    embeddings = encoder.encode(task.x.to(device), task.adj_t.to(device))[idx]
    kmeans = KMeans(n_clusters=n, mode='euclidean', verbose=0)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.centroids
    dist_matrix = euclidean_dist(centers, embeddings)
    q_idxs = [idx[i] for i in torch.argmin(dist_matrix, dim=1)]
    return q_idxs