import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

from src.se3_crossformer.se3_utils import (
    apply_direct_sum_W,
    equivariant_weight_matrix,
    RadialNetwork,
)
from src.se3_crossformer.attention import (
    IntraNeighborhoodAttention,
    InterNeighborhoodAttention,
    CrossAttention,
)
from src.se3_crossformer.spectral_partition import (
    spectral_partition,
    subgraph_center_of_mass,
    initial_message,
)

class SE3InterNeighborhoodLayer(nn.Module):
    def __init__(
        self,
        max_degree: int,
        feature_dim: int,     # channels per irrep degree (used as radial hidden width)
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.max_degree = max_degree
        self.feature_dim = feature_dim
        self.intra_attn = IntraNeighborhoodAttention(max_degree, feature_dim, hidden_dim)

        self.W_V_self_intra = nn.ModuleDict({
            str(l): nn.Linear(2 * l + 1, 2 * l + 1, bias=False)
            for l in range(max_degree + 1)
        })

        self.radial_V_intra: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_intra[key] = nn.ModuleDict({
                    str(J): RadialNetwork(num_basis=(2*l+1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.inter_attn = InterNeighborhoodAttention(max_degree, feature_dim, hidden_dim)

        self.W_V_self_msg = nn.ModuleDict({
            str(l): nn.Linear(2 * l + 1, 2 * l + 1, bias=False)
            for l in range(max_degree + 1)
        })

        self.radial_V_msg: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_msg[key] = nn.ModuleDict({
                    str(J): RadialNetwork(num_basis=(2*l+1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.cross_attn = CrossAttention(max_degree, feature_dim, hidden_dim)

        self.msg_to_phi: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.msg_to_phi[key] = nn.ModuleDict({
                    str(J): nn.Linear((2 * k + 1) * feature_dim, 1, bias=False)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.layer_norm = nn.ModuleDict({
            str(l): nn.LayerNorm(2 * l + 1)
            for l in range(max_degree + 1)
        })

    def _intra_update(
        self,
        f_in: Dict[int, torch.Tensor],   # {l: [N, 2l+1]}
        x: torch.Tensor,                  # [N, 3]
        neighbor_idx: torch.Tensor,       # [N, K]
        neighbor_mask: torch.Tensor,      # [N, K] bool
    ) -> Dict[int, torch.Tensor]:
        N, K = neighbor_idx.shape

        alpha = self.intra_attn(f_in, x, neighbor_idx, neighbor_mask)

        x_i   = x.unsqueeze(1).expand(N, K, 3)
        x_j   = x[neighbor_idx.view(-1)].view(N, K, 3)
        x_rel = (x_j - x_i).reshape(N * K, 3)              # [N*K, 3]

        f_j_flat: Dict[int, torch.Tensor] = {
            l: f_in[l][neighbor_idx.view(-1)]               # [N*K, 2l+1]
            for l in f_in
        }

        Wf_j_flat = apply_direct_sum_W(
            f=f_j_flat,
            x=x_rel,
            radial_nets=self.radial_V_intra,
            max_degree=self.max_degree,
        )

        f_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f_in:
                continue

            C = f_in[l].shape[1]
            self_term = self.W_V_self_intra[str(l)](f_in[l])   # [N, 2l+1]

            Wf_j = Wf_j_flat[l].view(N, K, C, 2 * l + 1)         # [N, K, C, 2l+1]
            alpha_exp = alpha.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)                  # [N, K, 1, 1]
            neighbor_term = (alpha_exp * Wf_j).sum(dim=1)       # [N, C, 2l+1]

            f_out[l] = self_term + neighbor_term

        return f_out

    def _message_update(
        self,
        m_in: Dict[int, torch.Tensor],    # {l: [S, C, 2l+1]}
        x_cm: torch.Tensor,               # [S, 3]
        subgraph_mask: torch.Tensor,      # [S, S] bool
    ) -> Dict[int, torch.Tensor]:
        S = x_cm.shape[0]
        C = m_in[0].shape[1]

        beta = self.inter_attn(m_in, x_cm, subgraph_mask)      # [S, S]

        x_cm_i = x_cm.unsqueeze(1).expand(S, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(S, S, 3)
        x_rel  = (x_cm_j - x_cm_i).reshape(S * S, 3)           # [S*S, 3]

        m_i_flat: Dict[int, torch.Tensor] = {
            l: m_in[l].unsqueeze(0).expand(S, S, C, 2 * l + 1).reshape(S * S, C, 2 * l + 1)
            for l in m_in
        }

        Wm_j_flat = apply_direct_sum_W(
            f=m_i_flat,
            x=x_rel,
            radial_nets=self.radial_V_msg,
            max_degree=self.max_degree,
        )

        m_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in m_in:
                continue

            self_term = self.W_V_self_msg[str(l)](m_in[l])     # [S, 2l+1]

            Wm_j = Wm_j_flat[l].view(S, S, C, 2 * l + 1)         # [S, S, C, 2l+1]
            beta_exp = beta.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)                      # [S, S, 1, 1]
            neighbor_term = (beta_exp * Wm_j).sum(dim=1)        # [S, C, 2l+1]

            m_out[l] = self_term + neighbor_term

        return m_out

    def _cross_update(
        self,
        f_out: Dict[int, torch.Tensor],   # {l: [N, C, 2l+1]}  from stage 1
        m_out: Dict[int, torch.Tensor],   # {l: [S, C, 2l+1]}  from stage 2
        x: torch.Tensor,                  # [N, 3]
        x_cm: torch.Tensor,               # [S, 3]
        node_to_subgraph: torch.Tensor,   # [N]
        subgraph_mask: torch.Tensor,      # [S, S]
    ) -> Dict[int, torch.Tensor]:
        N = x.shape[0]
        S = x_cm.shape[0]
        C = f_out[0].shape[1]

        gamma = self.cross_attn(
            f_out, m_out, x, x_cm, node_to_subgraph, subgraph_mask
        )

        x_i    = x.unsqueeze(1).expand(N, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(N, S, 3)
        x_rel  = (x_cm_j - x_i)                               # [N, S, 3]

        f_updated: Dict[int, torch.Tensor] = {}

        for l in f_out:
            feat = f_out[l]                                    # [N, 2l+1]
            cross_contrib = torch.zeros_like(feat)             # [N, 2l+1]

            for k in m_out:
                key = f"{l}_{k}"
                if key not in self.msg_to_phi:
                    continue

                m_k = m_out[k]                                 # [S, 2k+1]
                phi_nets = self.msg_to_phi[key]                # {J: Linear}

                m_k_expanded = m_k.unsqueeze(0).expand(N, S, C, 2 * k + 1)

                for b in range(S):
                    x_rel_b = x_rel[:, b, :]                   # [N, 3]
                    m_k_b   = m_k[b].flatten()                           # [C, 2k+1]

                    phi_b: Dict[int, torch.Tensor] = {}
                    for J_str, phi_net in phi_nets.items():
                        J = int(J_str)
                        phi_b[J] = phi_net(m_k_b).expand(N, 1)   # [N, 1]

                    class _FixedRadial(nn.Module):
                        def __init__(self, val):
                            super().__init__()
                            self._val = val
                        def forward(self, r, degree, k):
                            # r: [N]; return [N] precomputed phi
                            return self._val

                    radial_b = {str(J): _FixedRadial(phi_b[J]) for J in phi_b}

                    # W^{lk}(x_rel_b): [N, 2l+1, 2k+1]
                    W_b = equivariant_weight_matrix(
                        x=x_rel_b,
                        l=l,
                        k=k,
                        radial_fns=radial_b,
                    )

                    # Apply W_b to f_out^k: need f_out[k] [N, C, 2k+1]
                    if k not in f_out:
                        continue
                    f_k = f_out[k]                             # [N, C, 2k+1]

                    # [N, 2l+1, 2k+1] @ [N, C, 2k+1] -> [N, C, 2l+1]
                    Wf = torch.einsum("nij,ncj->nci", W_b, f_k)

                    cross_contrib = cross_contrib + gamma[:, b].sum(dim=-1).unsqueeze(-1).unsqueeze(-1) * Wf

            f_updated[l] = feat

        return f_updated

    def forward(
        self,
        f_in: Dict[int, torch.Tensor],    # {l: [N, 2l+1]}
        x: torch.Tensor,                  # [N, 3]
        neighbor_idx: torch.Tensor,        # [N, K]  intra-neighborhood indices
        neighbor_mask: torch.Tensor,       # [N, K]  bool
        x_cm: torch.Tensor,               # [S, 3]
        node_to_subgraph: torch.Tensor,   # [N]
        subgraph_mask: torch.Tensor,      # [S, S]  bool
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Returns (f_out, m_out).
        """
        f_out = self._intra_update(f_in, x, neighbor_idx, neighbor_mask)

        num_subgraphs = x_cm.shape[0]
        m_in  = initial_message(f_out, node_to_subgraph, num_subgraphs)
        m_out = self._message_update(m_in, x_cm, subgraph_mask)
        f_out = self._cross_update(f_out, m_out, x, x_cm, node_to_subgraph, subgraph_mask)

        return f_out, m_out
    
class SE3InterNeighborhoodTransformer(nn.Module):
    """
    Notes on extensions (not implemented):
        * Time-dependent MD: wrap this class in a spatio-temporal graph
          attention module that propagates across trajectory frames.
        * Batched molecules: add a batch index and use scatter-reduce for
          the final graph-level pooling.
        * IR spectra: return the degree-1 (vector) readout from f[1] and
          apply an FFT externally to obtain the power spectrum.
    """

    def __init__(
        self,
        in_features: int,
        max_degree: int = 2,
        num_layers: int = 4,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        num_parts: int = 4,         # number of spectral subgraphs
        out_dim: int = 19,           # output dimension (19 prediction targets)
        task: str = "regression",   # "regression" or "spectra"
    ):
        super().__init__()
        self.max_degree = max_degree
        self.num_parts  = num_parts
        self.task       = task

        self.input_embedding = nn.Linear(in_features, feature_dim)

        self.layers = nn.ModuleList([
            SE3InterNeighborhoodLayer(max_degree, feature_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.ModuleDict({
                str(l): nn.Sequential(
                    nn.Linear((2*l + 1) * feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, out_dim)
                )
                for l in range(max_degree + 1)
            })

    def _build_subgraph_info(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor,
        atomic_masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        node_to_subgraph = spectral_partition(
            edge_index, num_nodes, self.num_parts
        ).to(x.device)

        x_cm = subgraph_center_of_mass(
            x, atomic_masses, node_to_subgraph, self.num_parts
        )

        subgraph_mask = torch.ones(
            self.num_parts, self.num_parts, dtype=torch.bool, device=x.device
        )
        subgraph_mask.fill_diagonal_(False)

        return node_to_subgraph, x_cm, subgraph_mask

    def _build_neighbor_info(
        self,
        edge_index: torch.Tensor,
        node_to_subgraph: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device

        adj: list[list[int]] = [[] for _ in range(num_nodes)]
        for e in range(edge_index.shape[1]):
            i = edge_index[0, e].item()
            j = edge_index[1, e].item()
            if node_to_subgraph[i] == node_to_subgraph[j]:
                adj[i].append(j)

        # Ensure every node has at least one slot to avoid K=0 edge case
        K = max((len(a) for a in adj), default=1)

        neighbor_idx  = torch.zeros(num_nodes, K, dtype=torch.long,  device=device)
        neighbor_mask = torch.zeros(num_nodes, K, dtype=torch.bool,  device=device)

        for i, nbrs in enumerate(adj):
            for pos, j in enumerate(nbrs):
                neighbor_idx[i, pos]  = j
                neighbor_mask[i, pos] = True

        return neighbor_idx, neighbor_mask

    def forward(
        self,
        node_features: torch.Tensor,     # [N, d_in]
        x: torch.Tensor,                  # [N, 3]
        edge_index: torch.Tensor,         # [2, E]
        atomic_masses: torch.Tensor,      # [N]
    ) -> torch.Tensor:
        
        N = node_features.shape[0]

        f0 = self.input_embedding(node_features).unsqueeze(-1)   # [N, feature_dim]
    
        f1 = torch.randn(N, f0.shape[1], 3) * (1/math.sqrt(3.0))
        f2 = torch.randn(N, f0.shape[1], 5) * (1/math.sqrt(5.0))

        f: Dict[int, torch.Tensor] = {0: f0,
                                      1: f1,
                                      2: f2}

        node_to_subgraph, x_cm, subgraph_mask = self._build_subgraph_info(
            edge_index, N, x, atomic_masses
        )

        neighbor_idx, neighbor_mask = self._build_neighbor_info(
            edge_index, node_to_subgraph, N
        )

        for layer in self.layers:
            f, _ = layer(
                f_in=f,
                x=x,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                x_cm=x_cm,
                node_to_subgraph=node_to_subgraph,
                subgraph_mask=subgraph_mask,
            )

        scalar_features = f[0]                       # [N, C, 1]
        graph_embedding  = scalar_features.mean(0).flatten()   # [C]
        out = self.readout[str(0)](graph_embedding)           # [out_dim]

        return out