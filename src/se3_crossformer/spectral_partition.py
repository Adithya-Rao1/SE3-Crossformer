"""
spectral_partition.py  (optimized)
------------------------------------
Key changes vs. original
  • subgraph_center_of_mass: replaced the Python `for s in range(num_subgraphs)`
    loop with a single scatter_add over all nodes simultaneously.
    O(num_subgraphs) Python iterations → 2 GPU kernel launches.

  • initial_message: replaced the Python `for s in range(num_subgraphs)` loop
    with a vectorised scatter_mean over all nodes simultaneously.
    O(num_subgraphs × num_degrees) Python iterations → O(num_degrees) kernel
    launches (one scatter_mean per degree).

  • build_laplacian, spectral_partition, _kmeans_cluster: unchanged.
"""

import torch
import numpy as np
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Laplacian / spectral partition (unchanged)
# ---------------------------------------------------------------------------

def build_laplacian(
    edge_index:  torch.Tensor,
    num_nodes:   int,
    edge_weight: Optional[torch.Tensor] = None,
    normalized:  bool = True,
) -> torch.Tensor:
    """
    # TODO: For molecules with >200 atoms, switch to a sparse eigensolver.
    """
    row, col = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)

    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[row, col] = edge_weight

    if normalized:
        deg_inv_sqrt                    = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes, device=edge_index.device) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = torch.diag(deg) - A

    return L


def spectral_partition(
    edge_index:           torch.Tensor,
    num_nodes:            int,
    num_parts:            int,
    edge_weight:          Optional[torch.Tensor] = None,
    normalized_laplacian: bool = True,
) -> torch.Tensor:
    """
    Returns:
        node_to_subgraph: LongTensor[num_nodes], values in {0, ..., num_parts-1}
    """
    L = build_laplacian(edge_index, num_nodes, edge_weight, normalized_laplacian)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L.cpu())

        embedding1 = eigenvectors[:, 1 : num_parts + 1]      # [N, num_parts]
        norms      = embedding1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embedding  = embedding1 / norms

        node_to_subgraph = _kmeans_cluster(embedding.numpy(), num_parts)
    except Exception as e:
        print("Spectral partition error: ", e)
        node_to_subgraph = np.zeros(num_nodes, dtype=np.int64)

    return torch.tensor(node_to_subgraph, dtype=torch.long)


def _kmeans_cluster(embedding: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the k-means clustering step. "
            "Install with:  pip install scikit-learn"
        )
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    return kmeans.fit_predict(embedding)


# ---------------------------------------------------------------------------
# ① subgraph_center_of_mass — vectorised scatter, no Python loop
# ---------------------------------------------------------------------------

def subgraph_center_of_mass(
    x:                torch.Tensor,   # [N, 3]
    atomic_masses:    torch.Tensor,   # [N]
    node_to_subgraph: torch.Tensor,   # [N]
    num_subgraphs:    int,
) -> torch.Tensor:                    # [num_subgraphs, 3]
    """
    Compute per-subgraph centre of mass.

    Original used `for s in range(num_subgraphs)` → O(S) Python iterations.
    New implementation uses two scatter_add calls → 2 GPU kernel launches.
    """
    device, dtype = x.device, x.dtype

    # Weighted positions: [N, 3]
    m = atomic_masses.unsqueeze(1)                              # [N, 1]
    weighted_pos = m * x                                        # [N, 3]

    # Scatter-sum weighted positions and masses
    idx = node_to_subgraph.to(device)

    x_cm_num = torch.zeros(num_subgraphs, 3, device=device, dtype=dtype)
    x_cm_num.scatter_add_(
        0,
        idx.unsqueeze(1).expand(-1, 3),
        weighted_pos,
    )

    mass_total = torch.zeros(num_subgraphs, device=device, dtype=dtype)
    mass_total.scatter_add_(0, idx, atomic_masses)

    # Guard against empty subgraphs
    safe_mass = mass_total.clamp(min=1e-12).unsqueeze(1)       # [S, 1]

    return x_cm_num / safe_mass                                 # [S, 3]


# ---------------------------------------------------------------------------
# ② initial_message — vectorised scatter_mean, no Python loop
# ---------------------------------------------------------------------------

def initial_message(
    f_out:            Dict[int, torch.Tensor],   # {l: [N, C, 2l+1]}
    node_to_subgraph: torch.Tensor,              # [N]
    num_subgraphs:    int,
) -> Dict[int, torch.Tensor]:                    # {l: [S, C, 2l+1]}
    """
    Average node features per subgraph to form initial subgraph messages.

    Original used `for s in range(num_subgraphs)` → O(S) Python iterations
    per degree.  New implementation uses a single scatter_add + count divide
    per degree → 2 GPU kernel launches per degree instead of O(S).
    """
    m: Dict[int, torch.Tensor] = {}
    device = node_to_subgraph.device

    # Count nodes per subgraph for the mean normalisation
    counts = torch.zeros(num_subgraphs, device=device, dtype=torch.float32)
    counts.scatter_add_(
        0,
        node_to_subgraph,
        torch.ones(node_to_subgraph.shape[0], device=device),
    )
    safe_counts = counts.clamp(min=1.0)                          # [S]

    for l, feat in f_out.items():                               # feat: [N, C, 2l+1]
        N, C, D = feat.shape                                     # D = 2l+1
        m_l = torch.zeros(num_subgraphs, C, D, device=device, dtype=feat.dtype)

        # Expand index to [N, C, D]
        idx = node_to_subgraph.view(N, 1, 1).expand(N, C, D)

        m_l.scatter_add_(0, idx, feat)

        # Divide by count: [S, 1, 1]
        m_l = m_l / safe_counts.view(num_subgraphs, 1, 1)

        m[l] = m_l

    return m