from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import numpy as np

from src.se3_crossformer.spherical_harm import *
from src.se3_crossformer.irr_rep import x_to_alpha_beta
from src.se3_crossformer.bessel_gpu import BesselTable

from e3nn import o3

_CG_CACHE: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
_SH_BASIS_CHANGE_CACHE: Dict[int, torch.Tensor] = {}

def fit_sh_basis_change(l: int, n_samples: int = 20000, tol: float = 1e-3) -> torch.Tensor:
    if l in _SH_BASIS_CHANGE_CACHE:
        return _SH_BASIS_CHANGE_CACHE[l]

    from .irr_rep import x_to_alpha_beta

    n = torch.randn(n_samples, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    alpha, beta = x_to_alpha_beta(n)
    theta = math.pi - beta   

    Y_custom = get_spherical_harmonics(l, theta=theta, phi=alpha).to(torch.float64)                    
    Y_e3nn   = o3.spherical_harmonics(l, n, normalize=True, normalization='component').to(torch.float64)  

    U, *_ = torch.linalg.lstsq(Y_e3nn, Y_custom) 
    resid = (Y_e3nn @ U - Y_custom).abs().max().item()
    assert resid < tol, f"SH basis-change fit residual too high for l={l}: {resid:.2e}"

    Uo, _, Vt = torch.linalg.svd(U)
    U = (Uo @ Vt).to(torch.float32)

    _SH_BASIS_CHANGE_CACHE[l] = U
    return U

def clebsch_gordan_matrix(l: int, k: int) -> Dict[int, torch.Tensor]:
    if (l, k) in _CG_CACHE:
        return _CG_CACHE[(l, k)]

    U_l = fit_sh_basis_change(l).to(torch.float64)
    U_k = fit_sh_basis_change(k).to(torch.float64)

    result: Dict[int, torch.Tensor] = {}
    for J in range(abs(l - k), l + k + 1):
        U_J = fit_sh_basis_change(J).to(torch.float64)

        w3j     = o3.wigner_3j(l, k, J).to(torch.float64)                  
        Q_e3nn  = w3j.reshape((2 * l + 1) * (2 * k + 1), 2 * J + 1)        
        K = torch.kron(U_l, U_k)                                          
        Q_custom_J = K.T @ Q_e3nn @ U_J

        result[J] = Q_custom_J.to(torch.float32)

    _CG_CACHE[(l, k)] = result
    return result

def find_kth_sph_root(order, k, thresh=1e-8):
    """
    Finds kth root of spherical bessel function of the first kind of order n for -3 <= n <= 3
    """
    c = 0
    assert abs(order) in [0, 1, 2, 3], "Order out of bounds of function"

    if order == 0:
        c = (k+1) * math.pi
    elif  order in range(-3, 4):
        n = abs(order) + 1/2

        # Works for k \in {1, 2, 3}
        # beta = (k+1) * n + 1.85575 * math.pow(n, (1/3)) + 1.033

        guess = ((k+1) + abs(order)/2 - 1/4) * math.pi
        interval_begin = guess - math.pi/2
        interval_end = guess + math.pi/2

        while abs(guess) > thresh:
            c = (interval_begin + interval_end)/2
            guess = jv(n, c)
            if jv(n, interval_begin) * guess < 0:
                interval_end = c
            else:
                interval_begin = c

    return c

def spherical_bessel_first_kind(
    order: int,
    argument,  
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    if isinstance(argument, torch.Tensor):
        arg_np = argument.detach().cpu().numpy()
    elif isinstance(argument, np.ndarray):
        arg_np = argument
    else:
        arg_np = np.asarray(argument, dtype=np.float64)

    cyl   = jv(order + 0.5, arg_np)
    cyl_t = torch.as_tensor(cyl, dtype=torch.float32, device=device)

    safe   = cyl_t.abs().clamp(min=1e-12)
    mag    = (math.pi / 2.0) ** 0.5 / safe.sqrt()
    return torch.where(cyl_t >= 0, mag, -mag)

class RadialNetworkSFB(nn.Module):
    def __init__(self, num_basis: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.num_basis = num_basis
 
    def _basis(self, r: torch.Tensor, degree: int, k:int, cutoff_radius=3.0) -> torch.Tensor:
        bases = []

        for ord in range(-degree, degree + 1):
            kth_root  = find_kth_sph_root(ord, k)                 
            argument  = (kth_root / cutoff_radius) * r         

            sph_bes_denom = spherical_bessel_first_kind(ord + 1, kth_root, device='cpu')
            sph_bes_r     = spherical_bessel_first_kind(ord,     argument,  device='cpu')

            sph_bes_r = sph_bes_r.reshape(-1)  

            norm_factor = (
                2.0 / ((cutoff_radius ** 3) * (sph_bes_denom ** 2) * kth_root + 1e-12)
            ) ** 0.5

            e_sbf = (norm_factor * sph_bes_r)
            bases.append(e_sbf)
        
        return torch.stack(bases, dim=0).reshape(-1, 1)
 
    def forward(self, r: torch.Tensor, order: int, k: int) -> torch.Tensor:
        bases = self._basis(r, order, k)
        out = self.net(bases.to(r.device).reshape(-1, 2*order+1))
        return out

class RadialNetworkGSFB(nn.Module):
    _table_cache: Dict[Tuple[Tuple[int, ...], float, int, float], "BesselTable"] = {}
 
    def __init__(
        self,
        num_basis:     None = None,
        hidden_dim:    int = 32,
        cutoff_radius: float = 5.0,
        orders:        Tuple[int, ...] = (0, 1, 2, 3),
        N_cheb:        int = 64,
        seg_width:     float = 80.0,
        num_heads:     int = 1,
        edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.cutoff_radius = float(cutoff_radius)
        self.orders         = tuple(int(o) for o in orders)
        self.num_orders      = len(self.orders)
        self.num_heads       = num_heads
        self.edge_feature_dim = edge_feature_dim

        table = self._get_or_build_table(self.orders, self.cutoff_radius, N_cheb, seg_width)

        self.register_buffer("_coeffs", table.coeffs.clone())
        self.n_seg             = table.n_seg
        self.seg_width_actual  = table.seg_width_actual
        self.N_cheb             = table.N_cheb

        self.net = nn.Sequential(
            nn.Linear(self.num_orders + edge_feature_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )
 
    @classmethod
    def _get_or_build_table(cls, orders, cutoff_radius, N_cheb, seg_width) -> "BesselTable":
        key = (orders, float(cutoff_radius), int(N_cheb), float(seg_width))
        if key not in cls._table_cache:
            cls._table_cache[key] = BesselTable(
                list(orders), x_max=cutoff_radius,
                N_cheb=N_cheb, seg_width=seg_width, verbose=False,
            )
        return cls._table_cache[key]
 
    def _map_to_segments(self, r_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sw = self.seg_width_actual
        x       = r_flat.clamp(min=0.0, max=self.cutoff_radius - 1e-6)
        seg_idx = torch.floor(x / sw).clamp(min=0, max=self.n_seg - 1)
        a       = seg_idx * sw
        t_local = 2.0 * (x - a) / sw - 1.0
        return seg_idx.long(), t_local
 
    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        r_flat = r.reshape(-1).to(self._coeffs.dtype)
        seg_idx, t_local = self._map_to_segments(r_flat)
 
        c        = self._coeffs[:, seg_idx, :]                        
        theta    = torch.arccos(t_local.clamp(-1.0, 1.0))               
        k_idx    = torch.arange(self.N_cheb, device=r.device, dtype=self._coeffs.dtype)
        T_basis  = torch.cos(theta.unsqueeze(-1) * k_idx.unsqueeze(0)) 
        jl       = (c * T_basis.unsqueeze(0)).sum(dim=-1)               
        return jl.t().to(r.dtype)                                       
 
    def forward(self, r: torch.Tensor, edge_feat: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        basis = self._basis(r)
        if self.edge_feature_dim > 0:
            basis = torch.cat([basis, edge_feat], dim=-1)
        return self.net(basis)

class RadialNetworkGRBF(nn.Module):
    def __init__(
        self,
        num_basis:     int   = 8,
        hidden_dim:    int   = 32,
        cutoff_radius: float = 5.0,
        num_heads:     int = 1,
        edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.num_basis     = num_basis
        self.cutoff_radius = cutoff_radius
        self.num_heads     = num_heads
        self.edge_feature_dim = edge_feature_dim

        centres = torch.linspace(0.0, cutoff_radius, num_basis)
        self.register_buffer("centres", centres)

        width = cutoff_radius / max(num_basis - 1, 1)
        self.register_buffer(
            "inv_width_sq",
            torch.tensor(-1.0 / (2.0 * width ** 2)),
        )

        # f(r) = 0.5*(cos(pi*r/cutoff)+1) for r < cutoff, else 0
        self.net = nn.Sequential(
            nn.Linear(num_basis + edge_feature_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )

    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        diff    = r - self.centres.unsqueeze(0)          
        rbf     = torch.exp(self.inv_width_sq * diff * diff) 

        r_scaled = (r / self.cutoff_radius).clamp(max=1.0)         
        envelope = 0.5 * (torch.cos(math.pi * r_scaled) + 1.0)    

        return rbf * envelope   

    def forward(self, r: torch.Tensor, edge_feat: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        basis = self._basis(r)
        if self.edge_feature_dim > 0:
            basis = torch.cat([basis, edge_feat], dim=-1)
        return self.net(basis)

def _batched_linear_stack(
    nets:      list,
    seq_attr:  str,
    layer_idx: int,
    h:         torch.Tensor,
) -> torch.Tensor:
    layer0  = getattr(nets[0], seq_attr)[layer_idx]
    W_stack = torch.stack([getattr(n, seq_attr)[layer_idx].weight for n in nets], dim=0)  
    out = torch.einsum("moi,...mi->...mo", W_stack, h)
    if layer0.bias is not None:
        b_stack = torch.stack([getattr(n, seq_attr)[layer_idx].bias for n in nets], dim=0) 
        out = out + b_stack
    return out


def _batched_sequential_apply(nets: list, seq_attr: str, h: torch.Tensor) -> torch.Tensor:
    ref_seq = getattr(nets[0], seq_attr)
    for idx, layer in enumerate(ref_seq):
        if isinstance(layer, nn.Linear):
            h = _batched_linear_stack(nets, seq_attr, idx, h)
        else:
            h = layer(h)
    return h


def _batched_radial_forward(
    nets:      list,
    r:         torch.Tensor,
    edge_feat: Optional[torch.Tensor],
) -> torch.Tensor:
    M   = len(nets)
    ref = nets[0]

    basis = ref._basis(r)
    if ref.edge_feature_dim > 0:
        basis = torch.cat([basis, edge_feat], dim=-1)

    h = basis.unsqueeze(-2).expand(*basis.shape[:-1], M, basis.shape[-1])   
    return _batched_sequential_apply(nets, "net", h)   


def _batched_norm_readout_forward(nets: list, x: torch.Tensor) -> torch.Tensor:
    W_proj = torch.stack([n.equiv_proj.weight for n in nets], dim=0)   
    v_all  = torch.einsum("mhc,scd->smhd", W_proj, x)                  
    eps    = nets[0].eps
    norms  = torch.sqrt((v_all ** 2).sum(dim=-1) + eps)                
    out    = _batched_sequential_apply(nets, "mlp", norms)              
    return out.squeeze(-1)                                             


def apply_direct_sum_W(
    f:           Dict[int, torch.Tensor],
    x:           torch.Tensor,
    radial_nets: nn.ModuleDict,
    max_degree:  int,
    num_heads:   int = 1,
    edge_feat:   Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    from .irr_rep import x_to_alpha_beta

    device, dtype = x.device, x.dtype
    N = x.shape[0]

    out: Dict[int, torch.Tensor] = {}
    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C     = any_k.shape[-2]
        assert C % num_heads == 0, \
            f"channel count {C} not divisible by num_heads {num_heads}"
        out[l] = torch.zeros(
            *x.shape[:-1], C, 2 * l + 1, device=device, dtype=dtype,
        )

    if N == 0:
        return out

    r = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    alphas, betas = x_to_alpha_beta(x)

    groups: Dict[int, list] = {}
    cg_by_lk: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            cg = clebsch_gordan_matrix(l, k)
            cg_by_lk[(l, k)] = cg
            key = f"{l}_{k}"
            for J in range(abs(l - k), l + k + 1):
                if J not in cg or str(J) not in radial_nets[key]:
                    continue
                net = radial_nets[key][str(J)]
                in_dim = net.net[0].weight.shape[1]
                groups.setdefault(in_dim, []).append((l, k, J, net))

    phi_by_ljk: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for members in groups.values():
        nets = [m[3] for m in members]
        phi_batched = _batched_radial_forward(nets, r, edge_feat)   
        for i, (l, k, J, _net) in enumerate(members):
            phi_by_ljk[(l, k, J)] = phi_batched[:, i, :]            

    sh_cache: Dict[int, torch.Tensor] = {}
    def _sh(J: int) -> torch.Tensor:
        if J not in sh_cache:
            sh_cache[J] = get_spherical_harmonics(J, theta=(math.pi - betas), phi=alphas)
        return sh_cache[J]

    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C = any_k.shape[-2]
        head_dim = C // num_heads

        for k in range(max_degree + 1):
            key = f"{l}_{k}"
            any_net = next(iter(radial_nets[key].values()))
            H = any_net.num_heads
            cg = cg_by_lk[(l, k)]

            W = torch.zeros(N, H, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)
            for J in range(abs(l - k), l + k + 1):
                if (l, k, J) not in phi_by_ljk:
                    continue
                phi = phi_by_ljk[(l, k, J)]
                Q_J = cg[J].to(device=device, dtype=dtype)
                Y_J = _sh(J)
                QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
                had = phi.unsqueeze(-1) * QTY.unsqueeze(1)
                W   = W + had.view(N, H, 2 * l + 1, 2 * k + 1)

            f_k_headed = f[k].reshape(*f[k].shape[:-2], num_heads, head_dim, f[k].shape[-1])
            contrib = torch.einsum("...hij,...hcj->...hci", W, f_k_headed)
            contrib = contrib.reshape(*contrib.shape[:-3], C, 2 * l + 1)
            out[l]  = out[l] + contrib

    return out


def direct_sum_vmap(tensors):
    n1, n2    = tensors[0].shape[0], tensors[0].shape[1]
    flattened = [t.flatten(0, 1) for t in tensors]
    out_flat  = torch.vmap(torch.block_diag, in_dims=0)(*flattened)
    return out_flat.view(n1, n2, out_flat.shape[1], out_flat.shape[2])


def direct_sum_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    result = None
    for l in a:
        if l not in b:
            continue
        dot    = a[l].permute(0, 1, 3, 2) @ b[l]
        result = dot if result is None else direct_sum_vmap([result, dot])
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result


def softmax_over_neighbors(
    scores: torch.Tensor,
    mask:   torch.Tensor = None,
) -> torch.Tensor:
    if mask is not None:
        if mask.dim() < scores.dim():
            mask = mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(~mask, float(-1e9))
    dim = -2 if scores.dim() >= 3 else -1
    attn = torch.nn.functional.softmax(scores + 1e-8, dim=dim)
    return torch.nan_to_num(attn, nan=1e-9)


def verify_cg_orthogonality(max_degree: int = 2, tol: float = 1e-5) -> None:
    print("Verifying CG matrix orthogonality...")
    all_ok = True
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            cg     = clebsch_gordan_matrix(l, k)
            Q_full = torch.cat([cg[J] for J in sorted(cg.keys())], dim=1)
            QQT    = Q_full @ Q_full.T
            eye    = torch.eye(Q_full.shape[0], dtype=QQT.dtype)
            err    = (QQT - eye).abs().max().item()
            status = "OK" if err < tol else "FAIL"
            if err >= tol:
                all_ok = False
            print(f"  (l={l}, k={k}): max |QQ^T - I| = {err:.2e}  [{status}]")

    if all_ok:
        print("All CG matrices passed orthogonality check.")
    else:
        raise ValueError(
            "One or more CG matrices failed the orthogonality check."
        )

def fit_sh_to_cartesian(l: int, target_fn, n_samples: int = 20000, tol: float = 1e-4) -> torch.Tensor:
    n = torch.randn(n_samples, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    alpha, beta = x_to_alpha_beta(n)
    theta = math.pi - beta

    Y_l = get_spherical_harmonics(l, theta=theta, phi=alpha)   
    T   = target_fn(n)                                          

    M, *_ = torch.linalg.lstsq(Y_l, T)
    resid = (Y_l @ M - T).abs().max().item()
    assert resid < tol, f"SH{l}->Cartesian fit residual too high: {resid:.2e}"

    return M.T.float()

class IrrepLinear(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        x = x.transpose(-1,-2)
        x = self.weight(x)
        return x.transpose(-1,-2)

class EquivariantLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin))

    def forward(self, x):
        return torch.einsum("oc,...cm->...om", self.weight, x)

class EquivariantReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(C, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)     
        x = self.proj(x)             
        return x.squeeze(-1)        

class EquivariantNormReadout(nn.Module):
    def __init__(self, C: int, hidden: int = 64, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.equiv_proj = nn.Linear(C, hidden, bias=False)   
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.equiv_proj(x.transpose(-1, -2))              
        v = v.transpose(-1, -2)                               
        norms = torch.sqrt((v ** 2).sum(dim=-1) + self.eps)  
        return self.mlp(norms).squeeze(-1)                    

def cg_for_J(l: int, k: int, J: int) -> torch.Tensor:
    """Return the CG matrix for a single J from the cached dict."""
    from src.se3_crossformer.se3_utils import clebsch_gordan_matrix
    return clebsch_gordan_matrix(l, k)[J]

