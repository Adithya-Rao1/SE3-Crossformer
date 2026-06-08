import torch
import numpy as np
from typing import Optional, Tuple, Dict

def build_laplacian(
    edge_index: torch.Tensor,   # [2, E]
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,  # [E]
    normalized: bool = True,
) -> torch.Tensor:
    """
    # TODO: For molecules with >200 atoms, switch to a sparse eigensolver
    #       (e.g. scipy.sparse.linalg.eigsh via numpy bridge) to avoid O(N^2) memory.
    """
    row, col = edge_index[0], edge_index[1]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)

    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[row, col] = edge_weight

    if normalized:
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes, device=edge_index.device) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = torch.diag(deg) - A

    return L

def spectral_partition(
    edge_index: torch.Tensor,    # [2, E]
    num_nodes: int,
    num_parts: int,
    edge_weight: Optional[torch.Tensor] = None,
    normalized_laplacian: bool = True,
) -> torch.Tensor:
    """
    Steps:
      1. Build normalized Laplacian L.
      2. Compute the `num_parts` smallest eigenvectors (excluding the trivial one).
      3. Embed nodes into R^{num_parts} via the eigenvectors.
      4. Cluster with k-means.

    Returns:
        node_to_subgraph: LongTensor of shape [num_nodes], values in {0, ..., num_parts-1}

    # TODO: Replace the k-means step with the exact Elden multiway partitioning
    #       algorithm referenced in your manuscript. The current implementation
    #       uses standard spectral clustering as a placeholder.
    # TODO: Verify eigenvector sign convention and normalization used by Elden.
    # TODO: For disconnected graphs (some molecules may yield disconnected bond
    #       graphs), handle zero eigenvalues gracefully.
    """
    L = build_laplacian(edge_index, num_nodes, edge_weight, normalized_laplacian)

    eigenvalues, eigenvectors = torch.linalg.eigh(L.cpu())   # [N], [N, N]

    # Skip the trivial eigenvector (index 0, eigenvalue ~0)
    embedding = eigenvectors[:, 1 : num_parts + 1]           # [N, num_parts]

    norms = embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embedding = embedding / norms                             # [N, num_parts]

    # TODO: Replace with Elden-specific clustering.
    node_to_subgraph = _kmeans_cluster(embedding.numpy(), num_parts)

    return torch.tensor(node_to_subgraph, dtype=torch.long)


def _kmeans_cluster(embedding: np.ndarray, k: int) -> np.ndarray:
    """
    # TODO: Swap this for the Elden partitioning criterions.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the k-means clustering step. "
            "Install with:  pip install scikit-learn"
        )

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embedding)
    return labels

def subgraph_center_of_mass(
    x: torch.Tensor,                  # [N, 3]  atom positions
    atomic_masses: torch.Tensor,      # [N]     atomic masses (e.g. from periodic table)
    node_to_subgraph: torch.Tensor,   # [N]     subgraph assignment
    num_subgraphs: int,
) -> torch.Tensor:
    x_cm = torch.zeros(num_subgraphs, 3, device=x.device, dtype=x.dtype)
    mass_total = torch.zeros(num_subgraphs, device=x.device, dtype=x.dtype)

    for s in range(num_subgraphs):
        mask = node_to_subgraph == s              # [N] bool
        if mask.sum() == 0:
            continue
        m = atomic_masses[mask]                   # [N_s]
        pos = x[mask]                             # [N_s, 3]
        x_cm[s] = (m.unsqueeze(1) * pos).sum(0) / m.sum()
        mass_total[s] = m.sum()

    return x_cm


def initial_message(
    f_out: Dict,                       # {degree: [N, 2l+1]}  post-intra-attention features
    node_to_subgraph: torch.Tensor,    # [N]
    num_subgraphs: int,
) -> Dict:
    """
    TODO: The paper initialises m at l=1. Clarify whether m should be defined
          for all degrees or only l=1, and whether mean-pooling is the right
          aggregation (max, sum, attention-pool are alternatives).
    """
    from typing import Dict as D
    m: D = {}
    for l, feat in f_out.items():                   # feat: [N, C, 2l+1]
        m_l = torch.zeros(num_subgraphs, feat.shape[-2], feat.shape[-1], device=feat.device, dtype=feat.dtype)
        counts = torch.zeros(num_subgraphs, device=feat.device, dtype=feat.dtype)
        for s in range(num_subgraphs):
            mask = node_to_subgraph == s
            if mask.sum() > 0:
                # print(feat.shape, mask.shape)
                mask = mask.reshape(mask.shape[0], 1, 1)
                m_l[s] = (feat * mask).mean(0)
                counts[s] = mask.sum().float()
        m[l] = m_l
    return m

