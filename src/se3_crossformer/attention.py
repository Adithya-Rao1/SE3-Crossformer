import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.se3_crossformer.se3_utils import (
    apply_direct_sum_W,
    RadialNetwork,
    direct_sum_inner_product,
    softmax_over_neighbors,
)

class QKProjection(nn.Module):
    def __init__(self, max_degree: int, feature_dim: int, hidden_dim: int, batch_size=32):
        """
        Args:
            max_degree:  Maximum irrep degree L.
            feature_dim: Number of channels per degree (size of each irrep slice
                         is 2*l+1; feature_dim is used as the radial network width).
            hidden_dim:  Radial network hidden size.
        """
        super().__init__()
        self.max_degree = max_degree
        self.W_Q = nn.ModuleDict({
            str(l): nn.Linear(2 * l + 1, 2 * l + 1, bias=False)
            for l in range(max_degree + 1)
        })

        self.W_K_nets = nn.ModuleDict()   # unused placeholder (see radial_K)
        self.radial_K: Dict[str, nn.ModuleDict] = nn.ModuleDict()

        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                j_nets = nn.ModuleDict({
                    str(J): RadialNetwork(hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })
                self.radial_K[key] = j_nets

    # ------------------------------------------------------------------
    def query(self, f: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        q: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f:
                continue

            q[l] = self.W_Q[str(l)](f[l])   # [..., 2l+1]
        return q

    def key(
        self,
        f: Dict[int, torch.Tensor],
        x_rel: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        return apply_direct_sum_W(
            f=f,
            x=x_rel,
            weight_nets=self.W_K_nets,   # unused inside apply_direct_sum_W
            radial_nets=self.radial_K,   # {f"{l}_{k}": {J: RadialNetwork}}
            max_degree=self.max_degree,
        )

class IntraNeighborhoodAttention(nn.Module):
    def __init__(self, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(max_degree, feature_dim, hidden_dim)

        # Scaling factor: total dimension of direct-sum representation
        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],   # {degree: [N, 2l+1]}
        x: torch.Tensor,                  # [N, 3] absolute positions
        neighbor_idx: torch.Tensor,       # [N, K] indices of neighbors
        neighbor_mask: torch.Tensor,      # [N, K] bool mask
    ) -> torch.Tensor:
        N, K = neighbor_idx.shape

        q = self.qk.query(f_in)           # {l: [N, 2l+1]}

        f_j = {}

        for l in f_in:
            C = f_in[l].shape[1]

            gathered = f_in[l][neighbor_idx.reshape(-1)]

            f_j[l] = gathered.view(
                N,
                K,
                C,
                2*l+1
            )

        x_i = x.unsqueeze(1).expand(N, K, 3)             # [N, K, 3]
        x_j = x[neighbor_idx.view(-1)].view(N, K, 3)     # [N, K, 3]
        x_rel = x_j - x_i                                 # [N, K, 3]

        f_j_flat = {
            l: f_j[l].reshape(
                N*K,
                C,
                2*l+1
            )
        }
        x_rel_flat = x_rel.reshape(N * K, 3)

        k = self.qk.key(f_j_flat, x_rel_flat)            # {l: [N*K, 2l+1]}
        k = {l: k[l].view(N, K, C, 2 * l + 1) for l in k}

        q_expanded = {l: q[l].unsqueeze(1) for l in q.keys()}
        scores = direct_sum_inner_product(q_expanded, k) / self.scale   # [N, K]

        return softmax_over_neighbors(scores, neighbor_mask)             # [N, K]

class InterNeighborhoodAttention(nn.Module):
    def __init__(self, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree
        self.qk = QKProjection(max_degree, feature_dim, hidden_dim)

        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        m_in: Dict[int, torch.Tensor],    # {degree: [S, C, 2l+1]}  S = num subgraphs
        x_cm: torch.Tensor,               # [S, 3] center-of-mass positions
        subgraph_mask: torch.Tensor,      # [S, S] bool, True = valid pair
    ) -> torch.Tensor:

        S = x_cm.shape[0]

        q = self.qk.query(m_in)           # {l: [S, 2l+1]}

        x_cm_i = x_cm.unsqueeze(1).expand(S, S, 3)    # [S, S, 3]
        x_cm_j = x_cm.unsqueeze(0).expand(S, S, 3)    # [S, S, 3]
        x_rel  = x_cm_j - x_cm_i                       # [S, S, 3]

        C = m_in[0].shape[1] 

        m_j = {l: m_in[l].unsqueeze(0).expand(S, S, C, 2 * l + 1) for l in m_in}

        m_j_flat   = {l: m_j[l].reshape(S * S, C, 2 * l + 1) for l in m_j}
        x_rel_flat = x_rel.reshape(S * S, 3)

        k = self.qk.key(m_j_flat, x_rel_flat)          # {l: [S*S*C, 2l+1]}
        k = {l: k[l].view(S, S, C, 2 * l + 1) for l in k}

        q_expanded = {l: q[l].unsqueeze(1).expand(S, S, C, 2 * l + 1) for l in q}
        scores = direct_sum_inner_product(q_expanded, k) / self.scale   # [S, S]

        return softmax_over_neighbors(scores, subgraph_mask)

class CrossAttention(nn.Module):
    def __init__(self, max_degree: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.max_degree = max_degree

        self.qk_node = QKProjection(max_degree, feature_dim, hidden_dim)   # for Q
        self.qk_msg  = QKProjection(max_degree, feature_dim, hidden_dim)   # for K

        d = sum(2 * l + 1 for l in range(max_degree + 1))
        self.scale = d ** 0.5

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],    # {degree: [N, 2l+1]}  node features
        m_in: Dict[int, torch.Tensor],    # {degree: [S, 2l+1]}  subgraph messages
        x: torch.Tensor,                  # [N, 3]  node positions
        x_cm: torch.Tensor,               # [S, 3]  CM positions
        node_to_subgraph: torch.Tensor,   # [N] int, subgraph index for each node
        subgraph_mask: torch.Tensor,      # [S, S] bool  (True = valid subgraph pair)
    ) -> torch.Tensor:

        N = x.shape[0]
        S = x_cm.shape[0]
        C = f_in[0].shape[1]

        q = self.qk_node.query(f_in)     # {l: [N, 2l+1]}

        x_i    = x.unsqueeze(1).expand(N, S, 3)      # [N, S, 3]
        x_cm_j = x_cm.unsqueeze(0).expand(N, S, 3)   # [N, S, 3]
        x_rel  = x_cm_j - x_i                         # [N, S, 3]

        m_j = {l: m_in[l].unsqueeze(0).expand(N, S, C, 2 * l + 1) for l in m_in}

        m_j_flat   = {l: m_j[l].reshape(N * S, C, 2 * l + 1) for l in m_j}
        x_rel_flat = x_rel.reshape(N * S, 3)

        k = self.qk_msg.key(m_j_flat, x_rel_flat)    # {l: [N*S, 2l+1]}
        k = {l: k[l].view(N, S, C, 2 * l + 1) for l in k}

        q_expanded = {l: q[l].unsqueeze(1).expand(N, S, C, 2 * l + 1) for l in q}
        scores = direct_sum_inner_product(q_expanded, k) / self.scale   # [N, S]

        # Mask: node i must not attend to its own subgraph.
        # own_subgraph[i, s] = True  iff node i belongs to subgraph s.
        own_subgraph = (
            node_to_subgraph.unsqueeze(1)                         # [N, 1]
            == torch.arange(S, device=x.device).unsqueeze(0)     # [1, S]
        )                                                          # [N, S] bool

        # Expand the provided [S, S] validity mask to [N, S] by selecting
        # the row that corresponds to each node's own subgraph.
        cross_mask = subgraph_mask[node_to_subgraph]              # [N, S]
        # Additionally exclude the node's own subgraph from attention.
        cross_mask = cross_mask & ~own_subgraph                   # [N, S]

        return softmax_over_neighbors(scores, cross_mask)         # [N, S]