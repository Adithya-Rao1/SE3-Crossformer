import os

import torch
import torch_scatter
from torch_scatter import scatter_mean
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple
import math

from src.se3_crossformer.se3_utils import apply_direct_sum_W
from src.se3_crossformer.attention import IntraNeighborhoodAttention
from src.se3_crossformer.spherical_harm import get_spherical_harmonics
from src.se3_crossformer.se3_utils import *

def _cartesian_dipole(n: torch.Tensor) -> torch.Tensor:
    return n

def _cartesian_quadrupole(n: torch.Tensor) -> torch.Tensor:
    X, Y, Z = n[:, 0], n[:, 1], n[:, 2]
    Qxx = X * X - 1.0 / 3.0
    Qyy = Y * Y - 1.0 / 3.0
    Qxy = X * Y
    Qxz = X * Z
    Qyz = Y * Z
    return torch.stack([Qxx, Qxy, Qxz, Qyy, Qyz], dim=-1)

if os.path.exists("sh1_cartesian.pt"):
    SH1_TO_CARTESIAN = torch.load("sh1_cartesian.pt")
else:
    SH1_TO_CARTESIAN = fit_sh_to_cartesian(1, _cartesian_dipole)
    torch.save(SH1_TO_CARTESIAN, "sh1_cartesian.pt")

if os.path.exists("sh2_cartesian.pt"):
    SH2_TO_CARTESIAN = torch.load("sh2_cartesian.pt")
else:
    SH2_TO_CARTESIAN = fit_sh_to_cartesian(2, _cartesian_quadrupole)
    torch.save(SH2_TO_CARTESIAN, "sh2_cartesian.pt")

def _run_layer(layer, f_in, x, neighbor_idx, neighbor_mask, neighbor_bond_attr):
    return layer(
        f_in=f_in, x=x, neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask,
        neighbor_bond_attr=neighbor_bond_attr,
    )

class SE3BaseTransformer(nn.Module):
    def __init__(
        self,
        radial_net,
        in_features: int,
        max_degree:  int = 2,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        radius_cutoff: float = 5.0,
        scalar_out_dim:     int = 19, # 3 for polarizability tensor type-0 component
        task:        int = 0,
        use_bond_info:              bool = True,
        use_checkpointing:          bool = False,
    ):
        super().__init__()
        self.max_degree    = max_degree
        self.radius_cutoff = radius_cutoff
        self.task           = task
        self.use_checkpointing = use_checkpointing
        self.use_bond_info  = use_bond_info

        self.input_embedding = nn.Linear(in_features, feature_dim)

        if task == 0:
            self.readout = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, scalar_out_dim),
                )

        elif task == 1:
            self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN)
            self.readout = EquivariantReadout(feature_dim)

        elif task == 2:
            self.register_buffer("sh2_to_cartesian", SH2_TO_CARTESIAN)
            self.readout = EquivariantReadout(feature_dim)

        else:
            self.register_buffer("sh1_to_cartesian", SH1_TO_CARTESIAN)
            self.register_buffer("sh2_to_cartesian", SH2_TO_CARTESIAN)
            self.readout0 = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, scalar_out_dim),
                )
            self.readout1 = EquivariantReadout(feature_dim)
            self.readout2 = EquivariantReadout(feature_dim)

    def _build_radius_neighbor_info(
        self,
        x:          torch.Tensor, 
        num_nodes:  int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        if num_nodes <= 1:
            neighbor_idx  = torch.arange(num_nodes, device=device).unsqueeze(1)
            neighbor_mask = torch.zeros(num_nodes, 1, dtype=torch.bool, device=device)
            return neighbor_idx, neighbor_mask

        dist = torch.cdist(x, x)
        in_radius = (dist <= self.radius_cutoff) & ~torch.eye(
            num_nodes, dtype=torch.bool, device=device
        )
        src, dst = in_radius.nonzero(as_tuple=True)

        if src.numel() == 0:
            neighbor_idx  = torch.arange(num_nodes, device=device).unsqueeze(1)
            neighbor_mask = torch.zeros(num_nodes, 1, dtype=torch.bool, device=device)
            return neighbor_idx, neighbor_mask

        deg = torch.bincount(src, minlength=num_nodes)  
        K   = max(int(deg.max().item()), 1)

        neighbor_idx  = torch.zeros(num_nodes, K, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(num_nodes, K, dtype=torch.bool, device=device)

        slot = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for k in range(K):
            available = slot[src] == k
            s = src[available]
            d = dst[available]
            if s.numel() == 0:
                break
            neighbor_idx[s, k]  = d
            neighbor_mask[s, k] = True
            slot[s] += 1

        return neighbor_idx, neighbor_mask

    def _seed_equivariant_features(
        self,
        f0:            torch.Tensor,   
        x:             torch.Tensor,  
        atomic_masses: torch.Tensor,
        batch:         torch.Tensor,  
    ) -> Dict[int, torch.Tensor]:
        B = int(batch.max().item()) + 1

        m = atomic_masses.unsqueeze(-1)                                        
        mass_sum = torch_scatter.scatter_add(m, batch, dim=0, dim_size=B)    
        weighted_sum = torch_scatter.scatter_add(m * x, batch, dim=0, dim_size=B)  
        center = weighted_sum / mass_sum.clamp(min=1e-8)                     

        x_rel  = x - center[batch]                              

        r     = x_rel.norm(dim=-1).clamp(min=1e-8)             
        x_hat = x_rel / r.unsqueeze(-1)                         

        alpha, beta = x_to_alpha_beta(x_hat)
        theta = math.pi - beta

        Y1 = get_spherical_harmonics(1, theta=theta, phi=alpha)  
        Y2 = get_spherical_harmonics(2, theta=theta, phi=alpha)

        scalar = f0.squeeze(-1)         
        radial = r.unsqueeze(-1)         

        f1 = scalar.unsqueeze(-1) * (radial.unsqueeze(-1) * Y1.unsqueeze(1))  
        f2 = scalar.unsqueeze(-1) * (radial.unsqueeze(-1) * Y2.unsqueeze(1))  

        return {1: f1, 2: f2}

    def forward(
        self,
        node_features: torch.Tensor,
        x:             torch.Tensor,
        edge_index:    torch.Tensor,
        atomic_masses: torch.Tensor,
        batch:         torch.Tensor,
        edge_attr:     Optional[torch.Tensor] = None,
        vec_feat:      Optional[torch.tensor] = None,
        tensor_feat:   Optional[torch.Tensor] = None,
        graph_idx:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B = int(batch.max().item()) + 1

        f0 = self.input_embedding(node_features).unsqueeze(-1)
        seeded = self._seed_equivariant_features(f0, x, atomic_masses, batch)
        f: Dict[int, torch.Tensor] = {0: f0, 1: seeded[1], 2: seeded[2]}

        counts_per_graph = torch.bincount(batch, minlength=B)
        ptr = [0] + torch.cumsum(counts_per_graph, dim=0).tolist()

        num_bond_feats = edge_attr.shape[-1] if edge_attr is not None else 0

        per_graph = []
        K_global  = 0
        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g    = hi - lo

            pos_g  = x[lo:hi]
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g   = edge_index[:, mask_e] - lo
            edge_attr_g = edge_attr[mask_e] if edge_attr is not None else None

            nidx_local, nmask = self._build_radius_neighbor_info(pos_g, n_g)

            if self.use_bond_info and edge_attr_g is not None:
                bond_lookup = torch.zeros(
                    n_g, n_g, num_bond_feats, device=x.device, dtype=edge_attr_g.dtype
                )
                bond_lookup[ei_g[0], ei_g[1]] = edge_attr_g
                nbond_local = bond_lookup[
                    torch.arange(n_g, device=x.device).unsqueeze(1), nidx_local
                ]
                nbond_local = nbond_local * nmask.unsqueeze(-1).to(nbond_local.dtype)
            else:
                nbond_local = None

            per_graph.append((nidx_local + lo, nmask, nbond_local))
            K_global = max(K_global, nidx_local.shape[1])

        neighbor_idx_list:  list = []
        neighbor_mask_list: list = []
        neighbor_bond_list: list = []

        for nidx, nmask, nbond in per_graph:
            K_g = nidx.shape[1]
            if K_g < K_global:
                pad_idx  = torch.zeros(nidx.shape[0],  K_global - K_g,
                                       dtype=torch.long, device=x.device)
                pad_mask = torch.zeros(nmask.shape[0], K_global - K_g,
                                       dtype=torch.bool, device=x.device)
                nidx  = torch.cat([nidx,  pad_idx],  dim=1)
                nmask = torch.cat([nmask, pad_mask], dim=1)
                if nbond is not None:
                    pad_bond = torch.zeros(nbond.shape[0], K_global - K_g, num_bond_feats,
                                           dtype=nbond.dtype, device=x.device)
                    nbond = torch.cat([nbond, pad_bond], dim=1)
            neighbor_idx_list.append(nidx)
            neighbor_mask_list.append(nmask)
            if nbond is not None:
                neighbor_bond_list.append(nbond)

        neighbor_idx      = torch.cat(neighbor_idx_list,     dim=0)
        neighbor_mask     = torch.cat(neighbor_mask_list,    dim=0)
        neighbor_bond_attr = torch.cat(neighbor_bond_list, dim=0) if neighbor_bond_list else None

        if self.use_checkpointing and self.training:
            f, _ = checkpoint(
                _run_layer, self.layer, f, x, neighbor_idx, neighbor_mask,
                neighbor_bond_attr,
                use_reentrant=False,
            )
        else:
            f, _ = self.layer(
                f_in             = f,
                x                = x,
                neighbor_idx     = neighbor_idx,
                neighbor_mask    = neighbor_mask,
                neighbor_bond_attr = neighbor_bond_attr,
            )

        if self.task == 0:
            scalar_features  = f[0].squeeze(-1)                          
            graph_embeddings = scatter_mean(
                scalar_features, batch, dim=0, dim_size=B
            )                                                              
            out = self.readout(graph_embeddings)                       
            return out
        elif self.task == 1:
            vector_features = f[1]                                   
            graph_embeddings = scatter_mean(
                vector_features, batch, dim=0, dim_size=B
            )                                                          
            out_sh = self.readout(graph_embeddings)                       
            out    = torch.einsum('ij,bj->bi', self.sh1_to_cartesian, out_sh) 
            return out                        
        elif self.task == 2:
            tensor_features = f[2]                                        
            graph_embeddings = scatter_mean(
                tensor_features, batch, dim=0, dim_size=B
            )                                                                   
            out_sh = self.readout(graph_embeddings)                            
            out    = torch.einsum('ij,bj->bi', self.sh2_to_cartesian, out_sh)   
            return out
        else:
            scalar_features  = f[0].squeeze(-1)                          
            graph_embeddings0 = scatter_mean(
                scalar_features, batch, dim=0, dim_size=B
            )                                                              
            out0 = self.readout0(graph_embeddings0)

            vector_features = f[1]                                   
            graph_embeddings1 = scatter_mean(
                vector_features, batch, dim=0, dim_size=B
            )                                                          
            out1_sh = self.readout1(graph_embeddings1)                       
            out1    = torch.einsum('ij,bj->bi', self.sh1_to_cartesian, out1_sh)

            tensor_features = f[2]                                        
            graph_embeddings2 = scatter_mean(
                tensor_features, batch, dim=0, dim_size=B
            )                                                                   
            out2_sh = self.readout2(graph_embeddings2)                            
            out2    = torch.einsum('ij,bj->bi', self.sh2_to_cartesian, out2_sh)

            return out0, out1, out2

class SE3IntraOnlyLayer(nn.Module):
    def __init__(
        self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 64,
        bond_feature_dim: int = 0,
    ):
        super().__init__()
        self.max_degree  = max_degree
        self.feature_dim = feature_dim

        self.intra_attn = IntraNeighborhoodAttention(
            radial_net, max_degree, feature_dim, hidden_dim,
            edge_feature_dim=bond_feature_dim,
        )

        self.W_V_self_intra = nn.ModuleDict({
            str(l): IrrepLinear(feature_dim)
            for l in range(max_degree + 1)
        })
        
        self.radial_V_intra: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_intra[key] = nn.ModuleDict({
                    str(J): radial_net(
                        num_basis=(2 * l + 1), hidden_dim=hidden_dim,
                        num_heads=feature_dim, edge_feature_dim=bond_feature_dim,
                    )
                    for J in range(abs(l - k), l + k + 1)
                })

    def _intra_update(
        self,
        f_in:          Dict[int, torch.Tensor],
        x:             torch.Tensor,
        neighbor_idx:  torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_bond_attr: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        N, K = neighbor_idx.shape

        alpha = self.intra_attn(f_in, x, neighbor_idx, neighbor_mask, edge_attr=neighbor_bond_attr)  # (N, K)

        x_i   = x.unsqueeze(1).expand(N, K, 3)
        x_j   = x[neighbor_idx.view(-1)].view(N, K, 3)
        x_rel = (x_j - x_i).reshape(N * K, 3)

        f_j_flat: Dict[int, torch.Tensor] = {
            l: f_in[l][neighbor_idx.view(-1)] for l in f_in
        }

        bond_flat = neighbor_bond_attr.reshape(N * K, -1) if neighbor_bond_attr is not None else None

        Wf_j_flat = apply_direct_sum_W(
            f=f_j_flat, x=x_rel,
            radial_nets=self.radial_V_intra,
            max_degree=self.max_degree,
            num_heads=self.feature_dim,
            edge_feat=bond_flat,
        )

        f_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in f_in:
                continue
            C = f_in[l].shape[1]
            self_term     = self.W_V_self_intra[str(l)](f_in[l])
            Wf_j          = Wf_j_flat[l].view(N, K, C, 2 * l + 1)
            alpha_exp     = alpha.view(N, K, 1, 1)
            neighbor_term = (alpha_exp * Wf_j).sum(dim=1).reshape(N, C, 2 * l + 1)
            f_out[l]      = self_term + neighbor_term

        return f_out

    def forward(
        self,
        f_in:             Dict[int, torch.Tensor],
        x:                torch.Tensor,
        neighbor_idx:     torch.Tensor,
        neighbor_mask:    torch.Tensor,
        neighbor_bond_attr: Optional[torch.Tensor] = None,
    ):
        f_out = self._intra_update(f_in, x, neighbor_idx, neighbor_mask, neighbor_bond_attr=neighbor_bond_attr)
        m_out: Dict[int, torch.Tensor] = {}
        return f_out, m_out

class SE3IntraOnlyTransformer(SE3BaseTransformer):
    def __init__(
        self,
        radial_net,
        in_features: int,
        max_degree:  int = 2,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        radius_cutoff: float = 5.0,
        scalar_out_dim:     int = 19, # 3 for polarizability tensor type-0 component
        task:        int = 0,
        use_bond_info:              bool = True,
        bond_feature_dim: int = 0,
        use_checkpointing: bool = False,
    ):
        super().__init__(
            radial_net=radial_net,
            in_features=in_features,
            max_degree=max_degree,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            radius_cutoff=radius_cutoff,
            scalar_out_dim=scalar_out_dim,
            task=task,
            use_bond_info=use_bond_info,
            use_checkpointing=use_checkpointing,
        )

        self.layer = SE3IntraOnlyLayer(
            radial_net, max_degree, feature_dim, hidden_dim,
            bond_feature_dim=bond_feature_dim,
        )