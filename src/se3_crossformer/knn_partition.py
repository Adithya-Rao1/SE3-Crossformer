import torch
import numpy as np
from typing import Optional

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def build_knn_edge_index(
    x: torch.Tensor,     
    k: int = 10,
) -> torch.Tensor:        
    N = x.shape[0]
    k_eff = min(k, N - 1) if N > 1 else 0
    if k_eff == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    dist = torch.cdist(x, x)                                   
    dist.fill_diagonal_(float("inf"))
    _, knn_idx = torch.topk(dist, k_eff, dim=1, largest=False) 

    src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    dst = knn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def knn_partition(
    x:          torch.Tensor,   
    num_nodes:  int,
    num_parts:  int,
    k:          int = 10,
) -> torch.Tensor:             
    if num_nodes <= num_parts:
        return torch.arange(num_nodes, dtype=torch.long) % max(num_parts, 1)

    x_np = x.detach().cpu().numpy()
    k_eff = min(k, num_nodes - 1)

    try:
        connectivity = kneighbors_graph(
            x_np, n_neighbors=k_eff, mode="connectivity", include_self=False
        )
        connectivity = connectivity.maximum(connectivity.T)   

        clustering = AgglomerativeClustering(
            n_clusters=num_parts,
            connectivity=connectivity,
            linkage="ward",
        )
        node_to_subgraph = clustering.fit_predict(x_np)
    except Exception as e:
        print("KNN partition error: ", e)
        node_to_subgraph = np.zeros(num_nodes, dtype=np.int64)

    return torch.tensor(node_to_subgraph, dtype=torch.long)