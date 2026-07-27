import torch
import numpy as np
from typing import Optional, Dict, List

from sklearn.cluster import KMeans

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

DEFAULT_BOND_ORDER_WEIGHTS = [1.0, 2.0, 3.0, 1.5]

def bond_type_to_edge_weight(
    bond_type_onehot:   torch.Tensor,            
    bond_order_weights: Optional[List[float]] = None,
) -> torch.Tensor:                               
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
    bond_order: torch.Tensor,     
    power:      float = 1.0,
) -> torch.Tensor:               
    bond_order = bond_order.reshape(-1)
    if power != 1.0:
        return bond_order.pow(power)
    return bond_order


def node_connectivity_features(
    edge_index:    torch.Tensor,                   
    num_nodes:     int,
    bond_features: Optional[torch.Tensor] = None,  
) -> torch.Tensor:                               
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
    bond_type_onehot:          Optional[torch.Tensor] = None,   
    bond_order:                Optional[torch.Tensor] = None,   
    bond_order_weights:        Optional[List[float]] = None,
    bond_order_power:          float = 1.0,
    use_connectivity_features: bool = False,
) -> torch.Tensor:
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

        embedding1 = eigenvectors[:, 1 : num_parts + 1]      
        norms      = embedding1.norm(dim=1, keepdim=True).clamp(min=1e-8)
        embedding  = embedding1 / norms

        if use_connectivity_features:
            conn_feats = node_connectivity_features(
                edge_index, num_nodes, bond_features_for_conn
            ).cpu()
            conn_norm = conn_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
            conn_feats = conn_feats / conn_norm                
            embedding = torch.cat([embedding, conn_feats], dim=1)

        node_to_subgraph = _kmeans_cluster(embedding.numpy(), num_parts)
    except Exception as e:
        print("Spectral partition error: ", e)
        node_to_subgraph = np.zeros(num_nodes, dtype=np.int64)

    return torch.tensor(node_to_subgraph, dtype=torch.long)

def _kmeans_cluster(embedding: np.ndarray, k: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    return kmeans.fit_predict(embedding)

def subgraph_center_of_mass(
    x:                torch.Tensor,   
    atomic_masses:    torch.Tensor,   
    node_to_subgraph: torch.Tensor,   
    num_subgraphs:    int,
) -> torch.Tensor:                   
    device, dtype = x.device, x.dtype

    m = atomic_masses.unsqueeze(1)                             
    weighted_pos = m * x                                      

    idx = node_to_subgraph.to(device)

    x_cm_num = torch.zeros(num_subgraphs, 3, device=device, dtype=dtype)
    x_cm_num.scatter_add_(
        0,
        idx.unsqueeze(1).expand(-1, 3),
        weighted_pos,
    )

    mass_total = torch.zeros(num_subgraphs, device=device, dtype=dtype)
    mass_total.scatter_add_(0, idx, atomic_masses)

    safe_mass = mass_total.clamp(min=1e-12).unsqueeze(1)       

    return x_cm_num / safe_mass                                 

def initial_message(
    f_out:            Dict[int, torch.Tensor],   
    node_to_subgraph: torch.Tensor,             
    num_subgraphs:    int,
) -> Dict[int, torch.Tensor]:                    
    m: Dict[int, torch.Tensor] = {}
    device = node_to_subgraph.device

    counts = torch.zeros(num_subgraphs, device=device, dtype=torch.float32)
    counts.scatter_add_(
        0,
        node_to_subgraph,
        torch.ones(node_to_subgraph.shape[0], device=device),
    )
    safe_counts = counts.clamp(min=1.0)                         

    for l, feat in f_out.items():                               
        N, C, D = feat.shape                                 
        m_l = torch.zeros(num_subgraphs, C, D, device=device, dtype=feat.dtype)

        idx = node_to_subgraph.view(N, 1, 1).expand(N, C, D)

        m_l.scatter_add_(0, idx, feat)
        m_l = m_l / safe_counts.view(num_subgraphs, 1, 1)
        m[l] = m_l

    return m