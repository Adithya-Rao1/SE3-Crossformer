import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.se3_crossformer.se3_utils import (
    apply_direct_sum_W,
    softmax_over_neighbors,
)

class EquivariantLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin))

    def forward(self, x):
        # x: [N, Cin, 2l+1]
        return torch.einsum("oc,ncm->nom", self.weight, x)


class QKProjection(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int, batch_size=32):
        super().__init__()
        self.max_degree = max_degree
        self.W_Q = nn.ModuleDict({
            str(l): EquivariantLinear(cin=feature_dim, cout=feature_dim)
            for l in range(max_degree + 1)
        })

        self.radial_K: Dict[str, nn.ModuleDict] = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                j_nets = nn.ModuleDict({
                    str(J): radial_net(num_basis=2*l+1, hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })
                self.radial_K[key] = j_nets

    def query(self, f: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        q: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f:
                continue
            q[l] = self.W_Q[str(l)](f[l])   # [*, C, 2l+1]
        return q

    def key(
        self,
        f: Dict[int, torch.Tensor],
        x_rel: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        return apply_direct_sum_W(
            f=f,
            x=x_rel,
            radial_nets=self.radial_K,
            max_degree=self.max_degree,
        )


def _equivariant_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Scalar equivariant inner product: sum over degrees, channels, and irrep dims.

    a[l], b[l]: [..., C, 2l+1]
    Returns: [...] scalar per leading-dim pair.
    """
    result = None
    for l in a:
        if l not in b:
            continue
        # element-wise product then sum over last two dims (C and 2l+1)
        contrib = (a[l] * b[l]).sum(dim=(-2, -1))   # [...]
        result = contrib if result is None else result + contrib
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result


class IntraNeighborhoodAttention(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(radial_net, max_degree, feature_dim, hidden_dim)
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],   # {l: [N, C, 2l+1]}
        x: torch.Tensor,                  # [N, 3]
        neighbor_idx: torch.Tensor,       # [N, K]
        neighbor_mask: torch.Tensor,      # [N, K] bool, True = valid
    ) -> torch.Tensor:                    # [N, K]
        N, K = neighbor_idx.shape
        C = next(iter(f_in.values())).shape[1]

        q = self.qk.query(f_in)           # {l: [N, C, 2l+1]}

        # Gather neighbor features
        f_j_flat = {
            l: f_in[l][neighbor_idx.reshape(-1)]   # [N*K, C, 2l+1]
            for l in f_in
        }

        x_i        = x.unsqueeze(1).expand(N, K, 3)
        x_j        = x[neighbor_idx.view(-1)].view(N, K, 3)
        x_rel_flat = (x_j - x_i).reshape(N * K, 3)

        k_flat = self.qk.key(f_j_flat, x_rel_flat)   # {l: [N*K, C, 2l+1]}

        # Reshape q to [N*K, C, 2l+1] by repeating each node K times
        q_expanded = {
            l: q[l].unsqueeze(1).expand(N, K, C, 2*l+1).reshape(N*K, C, 2*l+1)
            for l in q
        }

        # Scalar score per (node, neighbor) pair → reshape to [N, K]
        scores = _equivariant_inner_product(q_expanded, k_flat) / self.scale
        scores = scores.view(N, K)   # [N, K]

        return softmax_over_neighbors(scores, neighbor_mask)   # [N, K]


class InterNeighborhoodAttention(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(radial_net, max_degree, feature_dim, hidden_dim)
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        m_in: Dict[int, torch.Tensor],    # {l: [S, C, 2l+1]}
        x_cm: torch.Tensor,               # [S, 3]
        subgraph_mask: torch.Tensor,      # [S, S] bool, True = valid
    ) -> torch.Tensor:                    # [S, S]
        S = x_cm.shape[0]
        C = next(iter(m_in.values())).shape[1]

        q = self.qk.query(m_in)   # {l: [S, C, 2l+1]}

        x_cm_i     = x_cm.unsqueeze(1).expand(S, S, 3)
        x_cm_j     = x_cm.unsqueeze(0).expand(S, S, 3)
        x_rel_flat = (x_cm_j - x_cm_i).reshape(S * S, 3)

        m_j_flat = {
            l: m_in[l].unsqueeze(0).expand(S, S, C, 2*l+1).reshape(S*S, C, 2*l+1)
            for l in m_in
        }

        k_flat = self.qk.key(m_j_flat, x_rel_flat)   # {l: [S*S, C, 2l+1]}

        # Repeat each subgraph query S times to pair with every key
        q_expanded = {
            l: q[l].unsqueeze(1).expand(S, S, C, 2*l+1).reshape(S*S, C, 2*l+1)
            for l in q
        }

        scores = _equivariant_inner_product(q_expanded, k_flat) / self.scale
        scores = scores.view(S, S)   # [S, S]

        return softmax_over_neighbors(scores, subgraph_mask)   # [S, S]


class CrossAttention(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk_node = QKProjection(radial_net, max_degree, feature_dim, hidden_dim)
        self.qk_msg  = QKProjection(radial_net, max_degree, feature_dim, hidden_dim)
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],    # {l: [N, C, 2l+1]}
        m_in: Dict[int, torch.Tensor],    # {l: [S, C, 2l+1]}
        x: torch.Tensor,                  # [N, 3]
        x_cm: torch.Tensor,               # [S, 3]
        node_to_subgraph: torch.Tensor,   # [N]
        subgraph_mask: torch.Tensor,      # [S, S] bool
    ) -> torch.Tensor:                    # [N, S]
        N = x.shape[0]
        S = x_cm.shape[0]
        C = next(iter(f_in.values())).shape[1]

        q = self.qk_node.query(f_in)   # {l: [N, C, 2l+1]}

        x_i        = x.unsqueeze(1).expand(N, S, 3)
        x_cm_j     = x_cm.unsqueeze(0).expand(N, S, 3)
        x_rel_flat = (x_cm_j - x_i).reshape(N * S, 3)

        m_j_flat = {
            l: m_in[l].unsqueeze(0).expand(N, S, C, 2*l+1).reshape(N*S, C, 2*l+1)
            for l in m_in
        }

        k_flat = self.qk_msg.key(m_j_flat, x_rel_flat)   # {l: [N*S, C, 2l+1]}

        q_expanded = {
            l: q[l].unsqueeze(1).expand(N, S, C, 2*l+1).reshape(N*S, C, 2*l+1)
            for l in q
        }

        scores = _equivariant_inner_product(q_expanded, k_flat) / self.scale
        scores = scores.view(N, S)   # [N, S]

        # Mask: exclude node's own subgraph
        own_subgraph = (
            node_to_subgraph.unsqueeze(1)
            == torch.arange(S, device=x.device).unsqueeze(0)
        )   # [N, S] bool — True where node belongs to subgraph s

        cross_mask = subgraph_mask[node_to_subgraph] & ~own_subgraph   # [N, S]

        return softmax_over_neighbors(scores, cross_mask)   # [N, S]