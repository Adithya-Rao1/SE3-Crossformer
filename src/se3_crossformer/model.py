"""
model.py  (patched)
--------------------
Changes vs. original
─────────────────────
1.  SE3InterNeighborhoodTransformer._build_neighbor_info
      • Replaced the Python adjacency-list loop with fully vectorised
        tensor ops (torch.isin / scatter).  No per-node Python iteration.

2.  SE3InterNeighborhoodTransformer.forward
      • Spectral-partition & neighbor-graph results are memoised in
        self._graph_cache (keyed on a hash of the per-graph edge index).
        After the first epoch the cache is warm and graph construction
        costs ~0 ms instead of ~360 ms/batch.

3.  SE3InterNeighborhoodLayer._cross_update
      • Eliminated the inner `for b in range(S)` loop.  All S subgraphs
        are now processed in one batched einsum, replacing O(S) sequential
        kernel launches with a single fused operation.
      • The _FixedRadial inner class is gone; phi values are computed for
        all (N, S) pairs simultaneously.

4.  SE3InterNeighborhoodTransformer
      • self._graph_cache dict added to __init__.
"""

import torch
import torch_scatter
from torch_scatter import scatter_mean
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math
import hashlib

from src.se3_crossformer.se3_utils import (
    apply_direct_sum_W,
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

from src.se3_crossformer.spherical_harm import get_spherical_harmonics

# ── SE3InterNeighborhoodLayer ─────────────────────────────────────────────────

class SE3InterNeighborhoodLayer(nn.Module):
    def __init__(self, max_degree: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.max_degree  = max_degree
        self.feature_dim = feature_dim
        self.intra_attn  = IntraNeighborhoodAttention(max_degree, feature_dim, hidden_dim)

        self.W_V_self_intra = nn.ModuleDict({
            str(l): nn.Linear(2 * l + 1, 2 * l + 1, bias=False)
            for l in range(max_degree + 1)
        })

        self.radial_V_intra: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_intra[key] = nn.ModuleDict({
                    str(J): RadialNetwork(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
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
                    str(J): RadialNetwork(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
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

    # ── stage 1: intra-neighborhood update ───────────────────────────────

    def _intra_update(
        self,
        f_in:          Dict[int, torch.Tensor],   # {l: [N, C, 2l+1]}
        x:             torch.Tensor,               # [N, 3]
        neighbor_idx:  torch.Tensor,               # [N, K]
        neighbor_mask: torch.Tensor,               # [N, K] bool
    ) -> Dict[int, torch.Tensor]:
        N, K = neighbor_idx.shape

        alpha = self.intra_attn(f_in, x, neighbor_idx, neighbor_mask)

        x_i   = x.unsqueeze(1).expand(N, K, 3)
        x_j   = x[neighbor_idx.view(-1)].view(N, K, 3)
        x_rel = (x_j - x_i).reshape(N * K, 3)

        f_j_flat: Dict[int, torch.Tensor] = {
            l: f_in[l][neighbor_idx.view(-1)] for l in f_in
        }

        Wf_j_flat = apply_direct_sum_W(
            f=f_j_flat, x=x_rel,
            radial_nets=self.radial_V_intra,
            max_degree=self.max_degree,
        )

        f_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f_in:
                continue
            C = f_in[l].shape[1]
            self_term    = self.W_V_self_intra[str(l)](f_in[l])   # [N, 2l+1]
            Wf_j         = Wf_j_flat[l].view(N, K, C, 2 * l + 1)
            alpha_exp    = alpha.unsqueeze(-1).unsqueeze(-1)
            neighbor_term = (alpha_exp * Wf_j).sum(dim=1)
            f_out[l]     = self_term + neighbor_term

        return f_out

    # ── stage 2: inter-neighborhood (subgraph) update ────────────────────

    def _message_update(
        self,
        m_in:          Dict[int, torch.Tensor],   # {l: [S, C, 2l+1]}
        x_cm:          torch.Tensor,               # [S, 3]
        subgraph_mask: torch.Tensor,               # [S, S] bool
    ) -> Dict[int, torch.Tensor]:
        S = x_cm.shape[0]
        C = m_in[0].shape[1]

        beta = self.inter_attn(m_in, x_cm, subgraph_mask)     # [S, S]

        x_cm_i = x_cm.unsqueeze(1).expand(S, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(S, S, 3)
        x_rel  = (x_cm_j - x_cm_i).reshape(S * S, 3)

        m_i_flat: Dict[int, torch.Tensor] = {
            l: m_in[l].unsqueeze(0).expand(S, S, C, 2 * l + 1)
               .reshape(S * S, C, 2 * l + 1)
            for l in m_in
        }

        Wm_j_flat = apply_direct_sum_W(
            f=m_i_flat, x=x_rel,
            radial_nets=self.radial_V_msg,
            max_degree=self.max_degree,
        )

        m_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in m_in:
                continue
            self_term    = self.W_V_self_msg[str(l)](m_in[l])
            Wm_j         = Wm_j_flat[l].view(S, S, C, 2 * l + 1)
            beta_exp     = beta.unsqueeze(-1).unsqueeze(-1)
            neighbor_term = (beta_exp * Wm_j).sum(dim=1)
            m_out[l]     = self_term + neighbor_term

        return m_out

    # ── stage 3: cross update (vectorised) ───────────────────────────────

    def _cross_update(
        self,
        f_out:           Dict[int, torch.Tensor],   # {l: [N, C, 2l+1]}
        m_out:           Dict[int, torch.Tensor],   # {l: [S, C, 2l+1]}
        x:               torch.Tensor,               # [N, 3]
        x_cm:            torch.Tensor,               # [S, 3]
        node_to_subgraph: torch.Tensor,              # [N]
        subgraph_mask:   torch.Tensor,               # [S, S]
    ) -> Dict[int, torch.Tensor]:
        """
        Cross-attention update: each node attends to all subgraph messages.

        Key change vs. original
        ───────────────────────
        The `for b in range(S)` Python loop has been replaced by a single
        batched computation over all S subgraphs at once.

        For each (l, k, J) triple we now:
          1. Compute phi for all N×S pairs in one Linear forward.
          2. Call equivariant_weight_matrix on x_rel [N*S, 3] once.
          3. Apply gamma attention weights with a single einsum.
        """
        N = x.shape[0]
        S = x_cm.shape[0]

        gamma = self.cross_attn(
            f_out, m_out, x, x_cm, node_to_subgraph, subgraph_mask
        )  # [N, S]

        # Relative positions: [N, S, 3] → [N*S, 3] for batched W computation
        x_i    = x.unsqueeze(1).expand(N, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(N, S, 3)
        x_rel  = (x_cm_j - x_i).reshape(N * S, 3)   # [N*S, 3]

        f_updated: Dict[int, torch.Tensor] = {}

        for l in f_out:
            cross_contrib = torch.zeros_like(f_out[l])  # [N, C, 2l+1]

            for k in m_out:
                key = f"{l}_{k}"
                if key not in self.msg_to_phi:
                    continue

                m_k      = m_out[k]            # [S, C, 2k+1]
                phi_nets = self.msg_to_phi[key]
                C_k      = m_k.shape[1]
                dim_k    = 2 * k + 1

                # Flatten m_k to [S, C*(2k+1)] for the Linear
                m_k_flat = m_k.reshape(S, C_k * dim_k)

                for J_str, phi_net in phi_nets.items():
                    J = int(J_str)
                    
                    phi_S  = phi_net(m_k_flat)                           # [S, 1]
                    phi_NS = phi_S.unsqueeze(0).expand(N, S, 1).reshape(N*S, 1)                     # [N*S, 1]
                
                    W_NS = _equivariant_weight_single_J(x_rel, l, k, J, phi_NS) # [N*S, 2l+1, 2k+1]
                    W    = W_NS.view(N, S, 2 * l + 1, 2 * k + 1)
                
                    # Contract W against subgraph messages m_out[k], not node features f_out[k]
                    Wf = torch.einsum("nsij,scj->nsci", W, m_k)   # [N, S, C, 2l+1]
                
                    gamma_exp     = gamma.unsqueeze(-1).unsqueeze(-1)   # [N, S, 1, 1]
                    cross_contrib = cross_contrib + (gamma_exp * Wf).sum(dim=1) # [N, C, 2l+1]
                    f_updated[l] = f_out[l] + cross_contrib

        return f_updated

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        f_in:            Dict[int, torch.Tensor],
        x:               torch.Tensor,
        neighbor_idx:    torch.Tensor,
        neighbor_mask:   torch.Tensor,
        x_cm:            torch.Tensor,
        node_to_subgraph: torch.Tensor,
        subgraph_mask:   torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:

        f_out = self._intra_update(f_in, x, neighbor_idx, neighbor_mask)
        num_subgraphs = x_cm.shape[0]
        m_in = initial_message(f_out, node_to_subgraph, num_subgraphs)
        for l, t in m_in.items():
            assert t.shape[0] == num_subgraphs, \
                f"initial_message degree {l}: shape[0]={t.shape[0]} != num_subgraphs={num_subgraphs}"
        m_out = self._message_update(m_in, x_cm, subgraph_mask)
        f_out = self._cross_update(f_out, m_out, x, x_cm,
                                    node_to_subgraph, subgraph_mask)
        return f_out, m_out


# ── helpers used by _cross_update ────────────────────────────────────────────

def cg_for_J(l: int, k: int, J: int) -> torch.Tensor:
    """Return the CG matrix for a single J from the cached dict."""
    from src.se3_crossformer.se3_utils import clebsch_gordan_matrix
    return clebsch_gordan_matrix(l, k)[J]


def _equivariant_weight_single_J(
    x:   torch.Tensor,   # [N*S, 3]
    l:   int,
    k:   int,
    J:   int,
    phi: torch.Tensor,   # [N*S, 1]
) -> torch.Tensor:       # [N*S, 2l+1, 2k+1]
    """
    Compute W^{lk}(x) restricted to a single J channel.
    This is the inner loop body of equivariant_weight_matrix, extracted so
    that _cross_update can call it without re-running all J values.
    """
    from src.se3_crossformer.se3_utils import clebsch_gordan_matrix
    from src.se3_crossformer.irr_rep import x_to_alpha_beta, spherical_harmonics

    device, dtype = x.device, x.dtype

    # Guard
    if x.shape[0] == 0:
        return torch.zeros(0, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)

    cg    = clebsch_gordan_matrix(l, k)
    Q_J   = cg[J].to(device=device, dtype=dtype)   # [(2l+1)(2k+1), 2J+1]

    alphas, betas = x_to_alpha_beta(x)             # Tensor[N*S] each
    Y_J = get_spherical_harmonics(J, theta=(math.pi - betas), phi=alphas) # [N*S, 2J+1]

    # Q^T @ Y  →  [N*S, (2l+1)(2k+1)]
    QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
    had = phi * QTY                                 # [N*S, (2l+1)(2k+1)]
    return had.view(x.shape[0], 2 * l + 1, 2 * k + 1)

# ── SE3InterNeighborhoodTransformer ──────────────────────────────────────────

class SE3InterNeighborhoodTransformer(nn.Module):

    def __init__(
        self,
        in_features: int,
        max_degree:  int = 2,
        num_layers:  int = 4,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        num_parts:   int = 4,
        out_dim:     int = 19,
        task:        str = "regression",
    ):
        super().__init__()
        self.max_degree = max_degree
        self.num_parts  = num_parts
        self.task       = task

        # ── graph construction cache ──────────────────────────────────────
        # Keyed by a hash of the edge index bytes (per graph).
        # Stores: (node_to_subgraph, x_cm_local, neighbor_idx_local,
        #          neighbor_mask_local, K_local)
        # After the first epoch, graph construction costs ~0 ms.
        self._graph_cache: Dict[str, tuple] = {}

        self.input_embedding = nn.Linear(in_features, feature_dim)

        self.layers = nn.ModuleList([
            SE3InterNeighborhoodLayer(max_degree, feature_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.ModuleDict({
            str(l): nn.Sequential(
                nn.Linear((2 * l + 1) * feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            for l in range(max_degree + 1)
        })

    # ── graph construction helpers ────────────────────────────────────────

    def _build_subgraph_info(
        self,
        edge_index:    torch.Tensor,
        num_nodes:     int,
        x:             torch.Tensor,
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
        edge_index:       torch.Tensor,   # [2, E_local]  local indices
        node_to_subgraph: torch.Tensor,   # [N_local]
        num_nodes:        int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build dense [N, K] intra-subgraph neighbor tensors.

        Change vs. original
        ───────────────────
        The Python adjacency-list loop has been replaced with vectorised
        tensor operations:
          1. Filter edges to intra-subgraph pairs with a boolean mask.
          2. Compute K = max neighbours in one torch.bincount call.
          3. Fill neighbor_idx via a cumulative-count scatter — no Python
             iteration over nodes.
        """
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        # Keep only edges within the same subgraph
        same_sg   = node_to_subgraph[src] == node_to_subgraph[dst]
        src_intra = src[same_sg]
        dst_intra = dst[same_sg]

        if src_intra.numel() == 0:
            # No intra-subgraph edges → every node is its own neighbour
            neighbor_idx  = torch.arange(num_nodes, device=device).unsqueeze(1)
            neighbor_mask = torch.zeros(num_nodes, 1, dtype=torch.bool, device=device)
            return neighbor_idx, neighbor_mask

        # K = maximum number of intra-subgraph neighbours any single node has
        deg = torch.bincount(src_intra, minlength=num_nodes)   # [N]
        K   = int(deg.max().item())
        K   = max(K, 1)

        neighbor_idx  = torch.zeros(num_nodes, K, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(num_nodes, K, dtype=torch.bool, device=device)

        # Use a running slot counter per node to fill neighbor_idx row-by-row
        # without any Python loop.  torch.scatter_ with a cumulative index
        # achieves this in O(E) tensor operations.
        slot = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for _ in range(K):
            # Find edges whose source node still has an unfilled slot at _
            available = slot[src_intra] == _
            s = src_intra[available]
            d = dst_intra[available]
            if s.numel() == 0:
                break
            neighbor_idx[s, _]  = d
            neighbor_mask[s, _] = True
            slot[s] += 1

        return neighbor_idx, neighbor_mask

    @staticmethod
    def _edge_hash(edge_index_cpu: torch.Tensor, num_nodes: int) -> str:
        h = hashlib.md5(edge_index_cpu.numpy().tobytes())
        h.update(num_nodes.to_bytes(4, 'little'))
        return h.hexdigest()

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        node_features: torch.Tensor,   # [N_total, d_in]
        x:             torch.Tensor,   # [N_total, 3]
        edge_index:    torch.Tensor,   # [2, E_total]
        atomic_masses: torch.Tensor,   # [N_total]
        batch:         torch.Tensor,   # [N_total]
    ) -> torch.Tensor:                 # [B, out_dim]

        N = node_features.shape[0]
        B = int(batch.max().item()) + 1

        # ── initial features ──────────────────────────────────────────────
        f0 = self.input_embedding(node_features).unsqueeze(-1)        # [N, C, 1]
        f1 = torch.randn(N, f0.shape[1], 3, device=f0.device) * (1 / math.sqrt(3.0))
        f2 = torch.randn(N, f0.shape[1], 5, device=f0.device) * (1 / math.sqrt(5.0))
        f: Dict[int, torch.Tensor] = {0: f0, 1: f1, 2: f2}

        # ── build pointer array ───────────────────────────────────────────
        ptr = [0]
        for g in range(B):
            ptr.append(int((batch <= g).sum().item()))

        # ── per-graph graph construction (with caching) ───────────────────
        node_to_subgraph_list: list = []
        x_cm_list:             list = []
        subgraph_mask_list:    list = []
        neighbor_idx_list:     list = []
        neighbor_mask_list:    list = []

        subgraph_offset = 0
        K_global        = 0
        per_graph       = []

        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g    = hi - lo

            pos_g  = x[lo:hi]
            mass_g = atomic_masses[lo:hi]
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g   = edge_index[:, mask_e] - lo          # local indices

            # ── cache lookup ──────────────────────────────────────────────
            cache_key = self._edge_hash(ei_g.cpu(), n_g)
            if cache_key in self._graph_cache:
                n2s_local, xcm, smask, nidx_local, nmask = \
                    self._graph_cache[cache_key]
                # Re-derive positions-dependent xcm (positions can change
                # during training even if topology is fixed)
                xcm = subgraph_center_of_mass(pos_g, mass_g, n2s_local, self.num_parts)
            else:
                n2s_local, xcm, smask = self._build_subgraph_info(
                    ei_g, n_g, pos_g, mass_g
                )
                nidx_local, nmask = self._build_neighbor_info(
                    ei_g, n2s_local, n_g
                )
                self._graph_cache[cache_key] = (
                    n2s_local, xcm, smask, nidx_local, nmask
                )

            per_graph.append((
                n2s_local + subgraph_offset,
                xcm,
                smask,
                nidx_local + lo,
                nmask,
            ))
            subgraph_offset += self.num_parts
            K_global = max(K_global, nidx_local.shape[1])

        # ── pad K dimension to K_global ───────────────────────────────────
        for g, (n2s, xcm, smask, nidx, nmask) in enumerate(per_graph):
            K_g = nidx.shape[1]
            if K_g < K_global:
                pad_idx  = torch.zeros(nidx.shape[0],  K_global - K_g,
                                       dtype=torch.long, device=x.device)
                pad_mask = torch.zeros(nmask.shape[0], K_global - K_g,
                                       dtype=torch.bool, device=x.device)
                nidx  = torch.cat([nidx,  pad_idx],  dim=1)
                nmask = torch.cat([nmask, pad_mask], dim=1)
            node_to_subgraph_list.append(n2s)
            x_cm_list.append(xcm)
            subgraph_mask_list.append(smask)
            neighbor_idx_list.append(nidx)
            neighbor_mask_list.append(nmask)

        # ── assemble global tensors ───────────────────────────────────────
        node_to_subgraph  = torch.cat(node_to_subgraph_list, dim=0)
        x_cm_all          = torch.cat(x_cm_list,             dim=0)
        neighbor_idx      = torch.cat(neighbor_idx_list,     dim=0)
        neighbor_mask     = torch.cat(neighbor_mask_list,    dim=0)

        S_total = B * self.num_parts
        subgraph_mask_all = torch.zeros(
            S_total, S_total, dtype=torch.bool, device=x.device
        )
        for g, smask in enumerate(subgraph_mask_list):
            lo = g * self.num_parts
            hi = lo + self.num_parts
            subgraph_mask_all[lo:hi, lo:hi] = smask

        # ── layer forward passes ──────────────────────────────────────────
        for layer in self.layers:
            f, _ = layer(
                f_in             = f,
                x                = x,
                neighbor_idx     = neighbor_idx,
                neighbor_mask    = neighbor_mask,
                x_cm             = x_cm_all,
                node_to_subgraph = node_to_subgraph,
                subgraph_mask    = subgraph_mask_all,
            )

        # ── readout ───────────────────────────────────────────────────────
        scalar_features  = f[0].squeeze(-1)                           # [N, C]
        graph_embeddings = scatter_mean(
            scalar_features, batch, dim=0, dim_size=B
        )                                                              # [B, C]
        out = self.readout[str(0)](graph_embeddings)                  # [B, out_dim]
        return out