import torch
import numpy as np
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# Laplacian / spectral partition
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

    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    deg.scatter_add_(0, row, edge_weight)

    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    A[row, col] = edge_weight

    if normalized:
        deg_inv_sqrt                    = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes, device=edge_index.device) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = torch.diag(deg) - A

    return L


# ---------------------------------------------------------------------------
# Molecular-information utilities (bond type + connectivity)
# ---------------------------------------------------------------------------

# Default physically-motivated bond-order weights, in the order
# [single, double, triple, aromatic]. These control how strongly the
# Laplacian pulls two bonded atoms into the same spectral cluster: a
# double/triple bond is a stiffer, more "rigid" coupling than a single bond,
# so it gets a larger weight. Override via `bond_order_weights` if your
# one-hot encoding uses a different channel order or includes extra bond
# types (e.g. dative, amide).
DEFAULT_BOND_ORDER_WEIGHTS = [1.0, 2.0, 3.0, 1.5]


def bond_type_to_edge_weight(
    bond_type_onehot:   torch.Tensor,             # [E, num_bond_types]
    bond_order_weights: Optional[List[float]] = None,
) -> torch.Tensor:                                # [E]
    """
    Convert one-hot bond-type encodings into a scalar per-edge weight.

    Note: `edge_index`/`bond_type_onehot` are assumed directed-both-ways
    (i.e. each bond appears as both (i, j) and (j, i)), matching the
    convention already used by `edge_index` elsewhere in this codebase.
    """
    num_types = bond_type_onehot.shape[-1]
    if bond_order_weights is None:
        if num_types > len(DEFAULT_BOND_ORDER_WEIGHTS):
            raise ValueError(
                f"bond_type_onehot has {num_types} bond types but only "
                f"{len(DEFAULT_BOND_ORDER_WEIGHTS)} default weights are "
                "defined. Pass `bond_order_weights` explicitly."
            )
        weights = torch.tensor(
            DEFAULT_BOND_ORDER_WEIGHTS[:num_types],
            device=bond_type_onehot.device,
            dtype=bond_type_onehot.dtype,
        )
    else:
        if len(bond_order_weights) != num_types:
            raise ValueError(
                f"bond_order_weights has {len(bond_order_weights)} entries "
                f"but bond_type_onehot has {num_types} bond types."
            )
        weights = torch.tensor(
            bond_order_weights,
            device=bond_type_onehot.device,
            dtype=bond_type_onehot.dtype,
        )
    return (bond_type_onehot * weights).sum(dim=-1)


def bond_order_edge_weight(
    bond_order: torch.Tensor,     # [E] or [E, 1] continuous bond order
    power:      float = 1.0,
) -> torch.Tensor:                # [E]
    """
    Turn a *continuous* bond order (e.g. RDKit's `Bond.GetBondTypeAsDouble()`,
    which already returns 1.0/2.0/3.0/1.5 for single/double/triple/aromatic)
    into an edge weight. This is the counterpart to `bond_type_to_edge_weight`
    for datasets that store bond order directly rather than as a one-hot
    category (e.g. QM9 loaded via RDKit, where `edge_attr` is the bond order).

    `power` lets you soften or sharpen the effect of bond order on the
    Laplacian (power=1.0 uses the order as-is; power<1.0 compresses the gap
    between single/double/triple; power=0.0 would recover the unweighted
    graph).
    """
    bond_order = bond_order.reshape(-1)
    if power != 1.0:
        return bond_order.pow(power)
    return bond_order


def node_connectivity_features(
    edge_index:    torch.Tensor,                   # [2, E]
    num_nodes:     int,
    bond_features: Optional[torch.Tensor] = None,  # [E] or [E, F]: one-hot bond
                                                     # types, continuous bond
                                                     # order, or any other
                                                     # per-edge molecular feature
) -> torch.Tensor:                                  # [N, 1 (+ F)]
    """
    Per-node molecular-connectivity descriptors:
      - degree (bond count)
      - summed per-edge molecular feature(s), if `bond_features` is given.
        This works whether `bond_features` is a one-hot bond-type encoding
        (giving per-bond-type incidence counts, e.g. "2 single bonds + 1
        aromatic bond") or a continuous bond order (giving a valence-like
        "total bond order" per atom).

    These get concatenated onto the spectral (eigenvector) embedding before
    k-means, so atoms with similar local bonding environments (e.g. two sp2
    ring carbons) are pulled together even when the unweighted topology
    alone wouldn't imply it.
    """
    device = edge_index.device
    row = edge_index[0]

    degree = torch.zeros(num_nodes, device=device)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    feats = [degree.unsqueeze(1)]

    if bond_features is not None:
        if bond_features.dim() == 1:
            bond_features = bond_features.unsqueeze(-1)
        num_feats = bond_features.shape[-1]
        feat_sums = torch.zeros(
            num_nodes, num_feats, device=device, dtype=bond_features.dtype
        )
        feat_sums.scatter_add_(
            0, row.unsqueeze(1).expand(-1, num_feats), bond_features
        )
        feats.append(feat_sums)

    return torch.cat(feats, dim=1)


def spectral_partition(
    edge_index:                torch.Tensor,
    num_nodes:                 int,
    num_parts:                 int,
    edge_weight:               Optional[torch.Tensor] = None,
    normalized_laplacian:      bool = True,
    bond_type_onehot:          Optional[torch.Tensor] = None,   # [E, num_bond_types]
    bond_order:                Optional[torch.Tensor] = None,   # [E] or [E, 1]
    bond_order_weights:        Optional[List[float]] = None,
    bond_order_power:          float = 1.0,
    use_connectivity_features: bool = False,
) -> torch.Tensor:
    """
    Returns:
        node_to_subgraph: LongTensor[num_nodes], values in {0, ..., num_parts-1}

    Molecular-information extensions
    ---------------------------------
    Pass exactly one of the following (or neither, for the original
    unweighted behavior) -- they're two encodings of the same idea and
    combining both would double-count bond order:

    bond_type_onehot: one-hot bond types (single/double/triple/aromatic,
        ...). Converted into edge weights via `bond_type_to_edge_weight`.
    bond_order: continuous bond order per edge, e.g. RDKit's
        `Bond.GetBondTypeAsDouble()` (1.0/2.0/3.0/1.5), which is exactly
        what `CustomQM9Dataset.edge_attr` stores. Converted into edge
        weights via `bond_order_edge_weight` (see `bond_order_power` to
        soften/sharpen its effect).

    Either way, the resulting bond weight shapes the Laplacian -- so bond
    order, not just raw topology, drives the spectral embedding. If
    `edge_weight` is *also* passed explicitly, it's multiplied with the
    bond weight (bond weight acts as a per-edge scale on top of whatever
    weighting you already had, e.g. inverse bond length).

    use_connectivity_features: if True, appends per-node connectivity
        descriptors (degree + summed bond feature, see
        `node_connectivity_features`) to the eigenvector embedding before
        k-means clustering.
    """
    if bond_type_onehot is not None and bond_order is not None:
        raise ValueError(
            "Pass either bond_type_onehot or bond_order, not both -- "
            "they're alternative encodings of the same bond information."
        )

    bond_features_for_conn: Optional[torch.Tensor] = None

    if bond_type_onehot is not None:
        bond_weight = bond_type_to_edge_weight(bond_type_onehot, bond_order_weights)
        edge_weight = bond_weight if edge_weight is None else edge_weight * bond_weight
        bond_features_for_conn = bond_type_onehot
    elif bond_order is not None:
        bond_weight = bond_order_edge_weight(bond_order, bond_order_power)
        edge_weight = bond_weight if edge_weight is None else edge_weight * bond_weight
        bond_features_for_conn = bond_weight.unsqueeze(-1)

    L = build_laplacian(edge_index, num_nodes, edge_weight, normalized_laplacian)

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L.cpu())

        embedding1 = eigenvectors[:, 1 : num_parts + 1]      # [N, num_parts]
        norms      = embedding1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embedding  = embedding1 / norms

        if use_connectivity_features:
            conn_feats = node_connectivity_features(
                edge_index, num_nodes, bond_features_for_conn
            ).cpu()
            conn_norm = conn_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
            conn_feats = conn_feats / conn_norm                # put on comparable scale
            embedding = torch.cat([embedding, conn_feats], dim=1)

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