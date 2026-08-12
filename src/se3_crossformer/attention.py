import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from src.se3_crossformer.se3_utils import apply_direct_sum_W, softmax_over_neighbors, EquivariantLinear

class QKProjection(nn.Module):
    def __init__(
        self,
        radial_net, max_degree: int, feature_dim: int, hidden_dim: int,
        batch_size=32, edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.max_degree  = max_degree
        self.feature_dim = feature_dim
        self.W_Q = nn.ModuleDict({
            str(l): EquivariantLinear(cin=feature_dim, cout=feature_dim)
            for l in range(max_degree + 1)
        })

        self.radial_K: Dict[str, nn.ModuleDict] = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                j_nets = nn.ModuleDict({
                    str(J): radial_net(
                        num_basis=2*l+1, hidden_dim=hidden_dim,
                        num_heads=feature_dim, edge_feature_dim=edge_feature_dim,
                    )
                    for J in range(abs(l - k), l + k + 1)
                })
                self.radial_K[key] = j_nets

    def query(self, f: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        q: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f:
                continue
            q[l] = self.W_Q[str(l)](f[l])
        return q

    def key(
        self,
        f: Dict[int, torch.Tensor],
        x_rel: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        return apply_direct_sum_W(
            f=f,
            x=x_rel,
            radial_nets=self.radial_K,
            max_degree=self.max_degree,
            num_heads=self.feature_dim,
            edge_feat=edge_feat,
        )

def _equivariant_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    result = None
    for l in a:
        if l not in b:
            continue
        contrib = (a[l] * b[l]).sum(dim=(-2, -1)) 
        result = contrib if result is None else result + contrib
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result

class IntraNeighborhoodAttention(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 32,
        edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(
            radial_net, max_degree, feature_dim, hidden_dim,
            edge_feature_dim=edge_feature_dim,
        )
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],
        x: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,   
    ) -> torch.Tensor:                              
        N, K = neighbor_idx.shape
        C = next(iter(f_in.values())).shape[1]

        q = self.qk.query(f_in)

        f_j_flat = {
            l: f_in[l][neighbor_idx.reshape(-1)]
            for l in f_in
        }

        x_i        = x.unsqueeze(1).expand(N, K, 3)
        x_j        = x[neighbor_idx.view(-1)].view(N, K, 3)
        x_rel_flat = (x_j - x_i).reshape(N * K, 3)

        edge_feat_flat = edge_attr.reshape(N * K, -1) if edge_attr is not None else None

        k_flat = self.qk.key(f_j_flat, x_rel_flat, edge_feat=edge_feat_flat)
        q_expanded = {
            l: q[l].unsqueeze(1).expand(N, K, C, 2*l+1).reshape(N*K, C, 2*l+1)
            for l in q
        }

        scores = _equivariant_inner_product(q_expanded, k_flat) / self.scale
        scores = scores.view(N, K)

        return softmax_over_neighbors(scores, neighbor_mask)

