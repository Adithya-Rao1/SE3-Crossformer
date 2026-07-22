import os

import torch
import torch_scatter
from torch_scatter import scatter_mean
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math
import hashlib

from src.se3_crossformer.se3_utils import apply_direct_sum_W
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
from src.se3_crossformer.knn_partition import knn_partition
from src.se3_crossformer.spherical_harm import get_spherical_harmonics
from src.se3_crossformer.se3_utils import clebsch_gordan_matrix
from src.se3_crossformer.irr_rep import x_to_alpha_beta

# Fixed change-of-basis: internal SH order (m=-1,0,+1) -> Cartesian (x,y,z).
_C1 = math.sqrt(3.0 / (4 * math.pi))
SH1_TO_CARTESIAN = torch.tensor([
    [ 0.,  0., -1.],   # x = -Y_{+1}/c
    [-1.,  0.,  0.],   # y = -Y_{-1}/c
    [ 0.,  1.,  0.],   # z =  Y_{0} /c
]) / _C1

def _cartesian_quadrupole(n: torch.Tensor) -> torch.Tensor:
    """
    n: [N, 3] unit vectors -> [N, 5] traceless symmetric quadrupole
    Q = n⊗n - I/3, independent components in order
    [Qxx, Qxy, Qxz, Qyy, Qyz]  (Qzz = -Qxx-Qyy is implied by tracelessness).
    """
    X, Y, Z = n[:, 0], n[:, 1], n[:, 2]
    Qxx = X * X - 1.0 / 3.0
    Qyy = Y * Y - 1.0 / 3.0
    Qxy = X * Y
    Qxz = X * Z
    Qyz = Y * Z
    return torch.stack([Qxx, Qxy, Qxz, Qyy, Qyz], dim=-1)

def fit_sh2_to_cartesian(n_samples: int = 20000, tol: float = 1e-4) -> torch.Tensor:
    """
    Fits the fixed linear map SH2 -> Cartesian-quadrupole by least squares
    """
    n = torch.randn(n_samples, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    alpha, beta = x_to_alpha_beta(n)
    theta = math.pi - beta
    Y2 = get_spherical_harmonics(2, theta=theta, phi=alpha)   
    T  = _cartesian_quadrupole(n)                             

    M, *_ = torch.linalg.lstsq(Y2, T)
    resid = (Y2 @ M - T).abs().max().item()
    assert resid < tol, f"SH2->Cartesian fit residual too high: {resid:.2e}"

    return M.T.float()  

if os.path.exists("sh2_cartesian.pt"):
    SH2_TO_CARTESIAN = torch.load("sh2_cartesian.pt")
else:
    SH2_TO_CARTESIAN = fit_sh2_to_cartesian()
    torch.save(SH2_TO_CARTESIAN, "sh2_cartesian.pt")

class IrrepLinear(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Linear(channels, channels)

    def forward(self, x):
        # x: [N,C,m]
        x = x.transpose(1,2)
        x = self.weight(x)
        return x.transpose(1,2)

class EquivariantReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64):
        super().__init__()
        self.ll = nn.Sequential(
            nn.Linear(C, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)   
        x = self.ll(x)          
        return x.squeeze(-1)   

def cg_for_J(l: int, k: int, J: int) -> torch.Tensor:
    """Return the CG matrix for a single J from the cached dict."""
    from src.se3_crossformer.se3_utils import clebsch_gordan_matrix
    return clebsch_gordan_matrix(l, k)[J]

def _equivariant_weight_single_J(
    x:   torch.Tensor,  
    l:   int,
    k:   int,
    J:   int,
    phi: torch.Tensor,  
) -> torch.Tensor:     

    device, dtype = x.device, x.dtype

    if x.shape[0] == 0:
        return torch.zeros(0, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)

    cg    = clebsch_gordan_matrix(l, k)
    Q_J   = cg[J].to(device=device, dtype=dtype) 

    alphas, betas = x_to_alpha_beta(x)            
    Y_J = get_spherical_harmonics(J, theta=(math.pi - betas), phi=alphas) 

    QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
    had = phi * QTY                               
    return had.view(x.shape[0], 2 * l + 1, 2 * k + 1)

class SE3BaseTransformer(nn.Module):
    def __init__(
        self,
        radial_net,
        in_features: int,
        max_degree:  int = 2,
        num_layers:  int = 4,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        num_parts:   int = 4,
        scalar_out_dim:     int = 19, # 3 for polarizability tensor type-0 component
        task:        int = 0,
        partition_type:             str = "spectral",  
        knn_k:                      int = 10,          
        use_bond_info:              bool = True,        
        bond_order_power:           float = 1.0,        
        use_connectivity_features:  bool = False,       
    ):
        super().__init__()
        self.max_degree = max_degree
        self.num_parts  = num_parts
        self.task       = task

        assert partition_type in ("spectral", "knn"), \
            f"partition_type must be 'spectral' or 'knn', got {partition_type!r}"
        self.partition_type            = partition_type
        self.knn_k                     = knn_k
        self.use_bond_info             = use_bond_info
        self.bond_order_power          = bond_order_power
        self.use_connectivity_features = use_connectivity_features

        self._graph_cache: Dict[str, tuple] = {}

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

    def _build_subgraph_info(
        self,
        edge_index:    torch.Tensor,
        num_nodes:     int,
        x:             torch.Tensor,
        atomic_masses: torch.Tensor,
        edge_attr:     Optional[torch.Tensor] = None,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.partition_type == "spectral":
            bond_order = edge_attr if (self.use_bond_info and edge_attr is not None) else None
            node_to_subgraph = spectral_partition(
                edge_index, num_nodes, self.num_parts,
                bond_order=bond_order,
                bond_order_power=self.bond_order_power,
                use_connectivity_features=self.use_connectivity_features,
            ).to(x.device)
        elif self.partition_type == "knn":
            node_to_subgraph = knn_partition(
                x, num_nodes, self.num_parts, k=self.knn_k
            ).to(x.device)
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type!r}")

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
        edge_index:       torch.Tensor,   
        node_to_subgraph: torch.Tensor,  
        num_nodes:        int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        src, dst = edge_index[0], edge_index[1]

        same_sg   = node_to_subgraph[src] == node_to_subgraph[dst]
        src_intra = src[same_sg]
        dst_intra = dst[same_sg]

        if src_intra.numel() == 0:
            neighbor_idx  = torch.arange(num_nodes, device=device).unsqueeze(1)
            neighbor_mask = torch.zeros(num_nodes, 1, dtype=torch.bool, device=device)
            return neighbor_idx, neighbor_mask

        deg = torch.bincount(src_intra, minlength=num_nodes)   # [N]
        K   = int(deg.max().item())
        K   = max(K, 1)

        neighbor_idx  = torch.zeros(num_nodes, K, dtype=torch.long, device=device)
        neighbor_mask = torch.zeros(num_nodes, K, dtype=torch.bool, device=device)

        slot = torch.zeros(num_nodes, dtype=torch.long, device=device)
        for _ in range(K):
            available = slot[src_intra] == _
            s = src_intra[available]
            d = dst_intra[available]
            if s.numel() == 0:
                break
            neighbor_idx[s, _]  = d
            neighbor_mask[s, _] = True
            slot[s] += 1

        return neighbor_idx, neighbor_mask

    def _partition_config_key(self) -> str:
        return "|".join([
            self.partition_type,
            str(self.knn_k),
            str(self.use_bond_info),
            str(self.bond_order_power),
            str(self.use_connectivity_features),
        ])

    @staticmethod
    def _edge_hash(edge_index_cpu: torch.Tensor, num_nodes: int, extra: str = "") -> str:
        h = hashlib.md5(edge_index_cpu.numpy().tobytes())
        h.update(num_nodes.to_bytes(4, 'little'))
        h.update(extra.encode("utf-8"))
        return h.hexdigest()
    
    def _seed_equivariant_features(
        self,
        f0:            torch.Tensor,   
        x:             torch.Tensor,  
        atomic_masses: torch.Tensor,
        batch:         torch.Tensor,  
    ) -> Dict[int, torch.Tensor]:
        B = int(batch.max().item()) + 1

        m = atomic_masses.unsqueeze(-1)                                        # [N, 1]
        mass_sum = torch_scatter.scatter_add(m, batch, dim=0, dim_size=B)      # [B, 1]
        weighted_sum = torch_scatter.scatter_add(m * x, batch, dim=0, dim_size=B)  # [B, 3]
        center = weighted_sum / mass_sum.clamp(min=1e-8)                        # [B, 3]

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
    ) -> torch.Tensor:               

        N = node_features.shape[0]
        B = int(batch.max().item()) + 1

        f0 = self.input_embedding(node_features).unsqueeze(-1)       
        seeded = self._seed_equivariant_features(f0, x, atomic_masses, batch)
        f: Dict[int, torch.Tensor] = {0: f0, 1: seeded[1], 2: seeded[2]}

        ptr = [0]
        for g in range(B):
            ptr.append(int((batch <= g).sum().item()))

        node_to_subgraph_list: list = []
        x_cm_list:             list = []
        subgraph_mask_list:    list = []
        neighbor_idx_list:     list = []
        neighbor_mask_list:    list = []

        subgraph_offset = 0
        K_global        = 0
        per_graph       = []
        partition_key   = self._partition_config_key()

        for g in range(B):
            lo, hi = ptr[g], ptr[g + 1]
            n_g    = hi - lo

            pos_g  = x[lo:hi]
            mass_g = atomic_masses[lo:hi]
            mask_e = (edge_index[0] >= lo) & (edge_index[0] < hi)
            ei_g   = edge_index[:, mask_e] - lo          
            edge_attr_g = edge_attr[mask_e] if edge_attr is not None else None

            cache_key = self._edge_hash(ei_g.cpu(), n_g, partition_key)
            if cache_key in self._graph_cache:
                n2s_local, xcm, smask, nidx_local, nmask = \
                    self._graph_cache[cache_key]
                xcm = subgraph_center_of_mass(pos_g, mass_g, n2s_local, self.num_parts)
            else:
                n2s_local, xcm, smask = self._build_subgraph_info(
                    ei_g, n_g, pos_g, mass_g, edge_attr=edge_attr_g
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

class SE3InterNeighborhoodLayer(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.max_degree  = max_degree
        self.feature_dim = feature_dim
        self.intra_attn  = IntraNeighborhoodAttention(radial_net, max_degree, feature_dim, hidden_dim)

        self.W_V_self_intra = nn.ModuleDict({
            str(l): IrrepLinear(feature_dim)
            for l in range(max_degree + 1)
        })

        self.radial_V_intra: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_intra[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.inter_attn = InterNeighborhoodAttention(radial_net, max_degree, feature_dim, hidden_dim)

        self.W_V_self_msg = nn.ModuleDict({
            str(l): IrrepLinear(feature_dim)
            for l in range(max_degree + 1)
        })

        self.radial_V_msg: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_msg[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

        self.cross_attn = CrossAttention(radial_net, max_degree, feature_dim, hidden_dim)

        self.msg_to_phi: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.msg_to_phi[key] = nn.ModuleDict({
                    str(J): EquivariantReadout(feature_dim)
                    for J in range(abs(l - k), l + k + 1)
                })

    def _intra_update(
        self,
        f_in:          Dict[int, torch.Tensor],  
        x:             torch.Tensor,              
        neighbor_idx:  torch.Tensor,             
        neighbor_mask: torch.Tensor,           
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
            self_term    = self.W_V_self_intra[str(l)](f_in[l]) 
            Wf_j         = Wf_j_flat[l].view(N, K, C, 2 * l + 1)
            alpha_exp    = alpha.unsqueeze(-1).unsqueeze(-1)
            neighbor_term = (alpha_exp * Wf_j).sum(dim=1)
            f_out[l]     = self_term + neighbor_term

        return f_out

    def _message_update(
        self,
        m_in:          Dict[int, torch.Tensor],   
        x_cm:          torch.Tensor,              
        subgraph_mask: torch.Tensor,              
    ) -> Dict[int, torch.Tensor]:
        S = x_cm.shape[0]
        C = m_in[0].shape[1]

        beta = self.inter_attn(m_in, x_cm, subgraph_mask)     

        x_cm_i = x_cm.unsqueeze(1).expand(S, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(S, S, 3)
        x_rel  = (x_cm_j - x_cm_i).reshape(S * S, 3)

        mask_flat = subgraph_mask.reshape(S * S)  

        m_j_flat: Dict[int, torch.Tensor] = {
            l: m_in[l].unsqueeze(0).expand(S, S, C, 2 * l + 1)
            .reshape(S * S, C, 2 * l + 1)
            for l in m_in
        }

        # Defining unit vector to prevent masked entried from affecting autograd graph of x_cm nor affecting m_out
        placeholder_dir = x_rel.new_tensor([0.0, 0.0, 1.0]).expand_as(x_rel)
        safe_x_rel = torch.where(
            mask_flat.unsqueeze(-1).expand_as(x_rel),
            x_rel,
            placeholder_dir,
        )

        Wm_j_flat = apply_direct_sum_W(
            f=m_j_flat, x=safe_x_rel,
            radial_nets=self.radial_V_msg,
            max_degree=self.max_degree,
        )

        mask_grid = subgraph_mask.unsqueeze(-1).unsqueeze(-1) 

        m_out: Dict[int, torch.Tensor] = {}
        for l in range(self.max_degree + 1):
            if l not in m_in:
                continue
            self_term     = self.W_V_self_msg[str(l)](m_in[l])
            Wm_j          = Wm_j_flat[l].view(S, S, C, 2 * l + 1)
            Wm_j          = torch.where(mask_grid, Wm_j, torch.zeros_like(Wm_j))
            beta_exp      = beta.unsqueeze(-1).unsqueeze(-1)
            neighbor_term = (beta_exp * Wm_j).sum(dim=1)
            m_out[l]      = self_term + neighbor_term

        return m_out

    def _cross_update(
        self,
        f_out:           Dict[int, torch.Tensor], 
        m_out:           Dict[int, torch.Tensor],  
        x:               torch.Tensor,              
        x_cm:            torch.Tensor,            
        node_to_subgraph: torch.Tensor,        
        subgraph_mask:   torch.Tensor,           
    ) -> Dict[int, torch.Tensor]:
        N = x.shape[0]
        S = x_cm.shape[0]

        gamma = self.cross_attn(
            f_out, m_out, x, x_cm, node_to_subgraph, subgraph_mask
        )  

        x_i    = x.unsqueeze(1).expand(N, S, 3)
        x_cm_j = x_cm.unsqueeze(0).expand(N, S, 3)
        x_rel  = (x_cm_j - x_i).reshape(N * S, 3)   

        f_updated: Dict[int, torch.Tensor] = {}

        for l in f_out:
            cross_contrib = torch.zeros_like(f_out[l])  

            for k in m_out:
                key = f"{l}_{k}"
                if key not in self.msg_to_phi:
                    continue

                m_k      = m_out[k]         
                phi_nets = self.msg_to_phi[key]
                C_k      = m_k.shape[1]
                dim_k    = 2 * k + 1

                # m_k_flat = m_k.reshape(S, C_k * dim_k)

                for J_str, phi_net in phi_nets.items():
                    J = int(J_str)
                    
                    phi_S  = phi_net(m_k)                           
                    phi_NS = phi_S.unsqueeze(0).expand(N, S, 1).reshape(N*S, 1)                     
                
                    W_NS = _equivariant_weight_single_J(x_rel, l, k, J, phi_NS) 
                    W    = W_NS.view(N, S, 2 * l + 1, 2 * k + 1)
                
                    Wf = torch.einsum("nsij,scj->nsci", W, m_k)   
                
                    gamma_exp     = gamma.unsqueeze(-1).unsqueeze(-1)   
                    cross_contrib = cross_contrib + (gamma_exp * Wf).sum(dim=1)
                    f_updated[l] = f_out[l] + cross_contrib

        return f_updated

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

class SE3InterNeighborhoodTransformer(SE3BaseTransformer):
    def __init__(
        self,
        radial_net, 
        in_features: int,
        max_degree:  int = 2,
        num_layers:  int = 4,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        num_parts:   int = 4,
        scalar_out_dim:     int = 19, # 3 for polarizability tensor type-0 component
        task:        int = 0,
        partition_type:             str = "spectral",
        knn_k:                      int = 10,
        use_bond_info:              bool = True,
        bond_order_power:           float = 1.0,
        use_connectivity_features:  bool = False,
    ):
        super().__init__(
            radial_net=radial_net,
            in_features=in_features,
            max_degree=max_degree,
            num_layers=num_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_parts=num_parts,
            scalar_out_dim=scalar_out_dim,
            task=task,
            partition_type=partition_type,
            knn_k=knn_k,
            use_bond_info=use_bond_info,
            bond_order_power=bond_order_power,
            use_connectivity_features=use_connectivity_features,
        )
        
        self.layers = nn.ModuleList([
            SE3InterNeighborhoodLayer(radial_net, max_degree, feature_dim, hidden_dim)
            for _ in range(num_layers)
        ])
 
class SE3IntraOnlyLayer(nn.Module):
    def __init__(self, radial_net, max_degree: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.max_degree  = max_degree
        self.feature_dim = feature_dim
 
        self.intra_attn = IntraNeighborhoodAttention(radial_net, max_degree, feature_dim, hidden_dim)
 
        self.W_V_self_intra = nn.ModuleDict({
            str(l): nn.Linear(2 * l + 1, 2 * l + 1, bias=False)
            for l in range(max_degree + 1)
        })
 
        self.radial_V_intra: nn.ModuleDict = nn.ModuleDict()
        for l in range(max_degree + 1):
            for k in range(max_degree + 1):
                key = f"{l}_{k}"
                self.radial_V_intra[key] = nn.ModuleDict({
                    str(J): radial_net(num_basis=(2 * l + 1), hidden_dim=hidden_dim)
                    for J in range(abs(l - k), l + k + 1)
                })
 
    def _intra_update(
        self,
        f_in:          Dict[int, torch.Tensor],   
        x:             torch.Tensor,               
        neighbor_idx:  torch.Tensor,             
        neighbor_mask: torch.Tensor,              
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
            self_term     = self.W_V_self_intra[str(l)](f_in[l])   
            Wf_j          = Wf_j_flat[l].view(N, K, C, 2 * l + 1)
            alpha_exp     = alpha.unsqueeze(-1).unsqueeze(-1)
            neighbor_term = (alpha_exp * Wf_j).sum(dim=1)
            f_out[l]      = self_term + neighbor_term
 
        return f_out
 
    def forward(
        self,
        f_in:             Dict[int, torch.Tensor],
        x:                torch.Tensor,
        neighbor_idx:     torch.Tensor,
        neighbor_mask:    torch.Tensor,
        x_cm:             torch.Tensor,            
        node_to_subgraph: torch.Tensor,           
        subgraph_mask:    torch.Tensor,            
    ):
        f_out = self._intra_update(f_in, x, neighbor_idx, neighbor_mask)
        m_out: Dict[int, torch.Tensor] = {}
        return f_out, m_out
  
class SE3IntraOnlyTransformer(SE3InterNeighborhoodTransformer):
    def __init__(
        self,
        radial_net, 
        in_features: int,
        max_degree:  int = 2,
        num_layers:  int = 4,
        feature_dim: int = 32,
        hidden_dim:  int = 64,
        num_parts:   int = 4,
        scalar_out_dim:     int = 19, # 3 for polarizability tensor type-0 component
        task:        int = 0,
        partition_type:             str = "spectral",
        knn_k:                      int = 10,
        use_bond_info:              bool = True,
        bond_order_power:           float = 1.0,
        use_connectivity_features:  bool = False,
    ):
        super().__init__(
            radial_net=radial_net,
            in_features=in_features,
            max_degree=max_degree,
            num_layers=num_layers,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_parts=num_parts,
            scalar_out_dim=scalar_out_dim,
            task=task,
            partition_type=partition_type,
            knn_k=knn_k,
            use_bond_info=use_bond_info,
            bond_order_power=bond_order_power,
            use_connectivity_features=use_connectivity_features,
        )

        self.layers = nn.ModuleList([
            SE3IntraOnlyLayer(radial_net, max_degree, feature_dim, hidden_dim)
            for _ in range(num_layers)
        ])