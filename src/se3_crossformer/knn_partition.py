"""
KNN-based atom clustering — ablation baseline for `spectral_partition`.

`spectral_partition.spectral_partition` embeds atoms via the eigenvectors of
the (bond-)graph Laplacian and then k-means-clusters that embedding. This
module instead builds a k-nearest-neighbor graph over atomic *spatial*
coordinates and clusters using KNN-connectivity-constrained agglomerative
clustering — no eigendecomposition, no bond-graph topology, just geometric
neighborhoods. It exists to answer the ablation question: is the spectral
eigen-embedding actually buying you anything over a much simpler,
purely-geometric neighbor graph?

The public entry point, `knn_partition`, mirrors the signature/return type
of `spectral_partition` (LongTensor[num_nodes] cluster assignment) so it can
be swapped in as a drop-in replacement in `_build_subgraph_info`.
"""

import torch
import numpy as np
from typing import Optional


def build_knn_edge_index(
    x: torch.Tensor,      # [N, 3]
    k: int = 10,
) -> torch.Tensor:        # [2, N*k]
    """
    Construct a directed, self-excluded spatial k-nearest-neighbor graph
    over atomic coordinates. Useful if you want to inspect/reuse the KNN
    graph itself (e.g. to feed into `neighbor_idx`/`neighbor_mask`
    construction elsewhere), separately from the clustering step below.
    """
    N = x.shape[0]
    k_eff = min(k, N - 1) if N > 1 else 0
    if k_eff == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)

    dist = torch.cdist(x, x)                                   # [N, N]
    dist.fill_diagonal_(float("inf"))
    _, knn_idx = torch.topk(dist, k_eff, dim=1, largest=False)  # [N, k_eff]

    src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    dst = knn_idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def knn_partition(
    x:          torch.Tensor,   # [N, 3]
    num_nodes:  int,
    num_parts:  int,
    k:          int = 10,
) -> torch.Tensor:              # LongTensor[num_nodes]
    """
    Partition atoms into `num_parts` subgraphs using KNN-connectivity
    constrained agglomerative clustering, as a simpler/cheaper alternative
    to `spectral_partition.spectral_partition`.

    Mechanics: build a k-NN graph on 3D coordinates, symmetrize it into an
    undirected connectivity constraint, then run Ward-linkage agglomerative
    clustering restricted to that connectivity (so clusters are always
    spatially contiguous under the KNN graph, similar in spirit to how
    spectral clustering is restricted by the bond-graph topology).

    Falls back to an all-zeros assignment on failure, matching the
    convention used by `spectral_partition`.

    Caveat: if `k` is too small, the KNN graph can fragment into more
    connected components than `num_parts`; sklearn will then return more
    clusters than requested. Increase `k` if you see this (a warning will
    be printed by sklearn).
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
    except ImportError:
        raise ImportError(
            "scikit-learn is required for KNN clustering. "
            "Install with:  pip install scikit-learn"
        )

    if num_nodes <= num_parts:
        # Degenerate case: not enough atoms to fill every part.
        return torch.arange(num_nodes, dtype=torch.long) % max(num_parts, 1)

    x_np = x.detach().cpu().numpy()
    k_eff = min(k, num_nodes - 1)

    try:
        connectivity = kneighbors_graph(
            x_np, n_neighbors=k_eff, mode="connectivity", include_self=False
        )
        connectivity = connectivity.maximum(connectivity.T)   # symmetrize

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