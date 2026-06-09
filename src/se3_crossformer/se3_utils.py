import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple

from src.se3_crossformer.irr_rep import *

from scipy.special import jv
import math

def find_kth_sph_root(order, k, thresh=1e-8):
    """
    Finds kth root of spherical bessel function of the first kind of order n for -3 <= n <= 3
    """
    c = 0
    assert abs(order) in [0, 1, 2, 3], "Order out of bounds of function"

    if order == 0:
        c = k * math.pi
    elif  order in [-3, -2, -1, 1, 2, 3]:
        """
        Using absolute order since J_(-n)(x) = (-1)^n * J_n(x)
        So,
        J_(-n)(x) and J_n(x) have same roots
        """
        n = abs(order) + 1/2

        # Works for k \in {1, 2, 3}
        beta = k * n + 1.85575 * math.pow(n, (1/3)) + 1.033
        
        interval_begin = beta - math.pi/2
        interval_end = beta + math.pi/2

        guess = 1.0
        while abs(guess) > thresh:
            c = (interval_begin + interval_end)/2
            guess = jv(n, c)

            """
            If c < 0 and begin < 0, zero between c and end (+)

            If c > 0 and begin < 0, zero between begin and c (-)

            If c < 0 and begin > 0, zero between begin and c (-)

            If c > 0 and begin > 0, zero between c and end (+)
            """
            if jv(n, interval_begin) * guess < 0: # root in [init_begin, c]
                interval_end = c
            else:
                interval_begin = c # root in [c, init_end]
                    
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

# Module-level cache: (l, k) -> {J: tensor[(2l+1)(2k+1), 2J+1]}
_CG_CACHE: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
 
 
def clebsch_gordan_matrix(l: int, k: int) -> Dict[int, torch.Tensor]:
    if (l, k) in _CG_CACHE:
        return _CG_CACHE[(l, k)]
 
    try:
        from sympy.physics.quantum.cg import CG
        from sympy import Rational, sqrt as sympy_sqrt, N as sympy_N
    except ImportError:
        raise ImportError(
            "sympy is required for Clebsch-Gordan computation.\n"
            "Install with:  pip install sympy"
        )
 
    dim_out = (2 * l + 1) * (2 * k + 1)   # rows of Q^{lk}
 
    result: Dict[int, torch.Tensor] = {}
 
    for J in range(abs(l - k), l + k + 1):
        dim_J = 2 * J + 1
        # Q_J has shape [dim_out, dim_J]
        # Row index:  (m_l + l) * (2k+1) + (m_k + k)    (matches vec convention)
        # Col index:  M + J
        Q_J = torch.zeros(dim_out, dim_J, dtype=torch.float64)
 
        for m_l in range(-l, l + 1):
            for m_k in range(-k, k + 1):
                row = (m_l + l) * (2 * k + 1) + (m_k + k)
                for M in range(-J, J + 1):
                    col = M + J
                    # sympy CG(j1, m1, j2, m2, J, M).doit()
                    # <l, m_l ; k, m_k | J, M>
                    cg_val = CG(
                        Rational(l),  Rational(m_l),
                        Rational(k),  Rational(m_k),
                        Rational(J),  Rational(M),
                    ).doit()
                    Q_J[row, col] = float(sympy_N(cg_val))
 
        result[J] = Q_J.to(torch.float32)
 
    _CG_CACHE[(l, k)] = result
    return result

# verify_cg_orthogonality() --> All CG matrices passed orthogonality check.

def precompute_cg_matrices(max_degree: int) -> None:
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            clebsch_gordan_matrix(l, k)
    print(f"Precomputed CG matrices for degrees 0..{max_degree}.")
 
def equivariant_weight_matrix(
    x: torch.Tensor,
    l: int,
    k: int,
    radial_fns: Dict[int, nn.Module],
) -> torch.Tensor:
    
    batch_shape = x.shape[:-1]
    device, dtype = x.device, x.dtype
 
    r = x.norm(dim=-1).clamp(min=1e-8)   # [...], scalar distance
 
    cg = clebsch_gordan_matrix(l, k)      # {J: [(2l+1)(2k+1), 2J+1]}
 
    W = torch.zeros(*batch_shape, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype) # [N, 2l+1, 2k+1]
 
    for J, Q_J in cg.items():
        # Q_J: [(2l+1)(2k+1), 2J+1], move to correct device/dtype
        Q_J = Q_J.to(device=device, dtype=dtype)
 
        # Y_J(x): [..., 2J+1]
        alphas, betas = x_to_alpha_beta(x)
        Y_J = []
        for alpha, beta in zip(alphas, betas):
            Y_J.append(spherical_harmonics(J, alpha, beta))
        Y_J = torch.tensor(Y_J, device=device).unsqueeze(-1)

        phi_J = radial_fns[J](r)          # [N]
 
        # Q_J^T @ Y_J  ->  [..., (2l+1)(2k+1)]
        # Y_J: [..., 2J+1], Q_J.T: [2J+1, (2l+1)(2k+1)]
        # einsum: 'ji,...i->...j'  with j = (2l+1)(2k+1), i = 2J+1
        QTY = torch.einsum("ji,...i->...j", Q_J, Y_J)   # [..., (2l+1)(2k+1)]
 
        # Reshape to [..., 2l+1, 2k+1] and weight by phi_J
        QTY_mat = QTY.view(-1, 2 * l + 1, 2 * k + 1)

        W = W + phi_J.unsqueeze(-1).unsqueeze(-1) * QTY_mat
 
    return W
 
class RadialNetwork(nn.Module):
    def __init__(self, num_basis: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_basis = num_basis
 
    def _basis(self, r: torch.Tensor, order: int, k:int, cutoff_radius=3.0) -> torch.Tensor:
        # TODO: Verify basis choice aligns with the rest of the architecture.
        bases = []

        for ord in range(-order, order + 1):
            kth_root  = find_kth_sph_root(ord, k)                 
            argument  = (kth_root / cutoff_radius) * r         

            sph_bes_denom = spherical_bessel_first_kind(ord + 1, kth_root, device='cpu')
            sph_bes_r     = spherical_bessel_first_kind(ord,     argument,  device='cpu')

            sph_bes_r = sph_bes_r.reshape(-1)  # [N_atoms]

            norm_factor = (
                2.0 / ((cutoff_radius ** 3) * (sph_bes_denom ** 2) * kth_root + 1e-12)
            ) ** 0.5

            e_sbf = (norm_factor * sph_bes_r).unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)
            bases.append(e_sbf)
        
        return torch.stack(bases, dim=0).squeeze(0)
 
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r: [...], returns scalar [...] """
        return self.net(self._basis(r)).squeeze(-1)
 
def apply_direct_sum_W(
    f: Dict[int, torch.Tensor],
    x: torch.Tensor,
    weight_nets: nn.ModuleDict,
    radial_nets: nn.ModuleDict,
    max_degree: int,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}

    for l in range(max_degree + 1):
        # infer channel dimension from any existing k
        any_k = next(iter(f.values()))
        C = any_k.shape[-2]

        out[l] = torch.zeros(
            *x.shape[:-1],
            C,
            2 * l + 1,
            device=x.device,
            dtype=x.dtype,
        )

        for k in range(max_degree + 1):
            if k not in f:
                continue

            key = f"{l}_{k}"
            if key not in weight_nets:
                continue

            W = equivariant_weight_matrix(x, l, k, radial_nets[key])
            # W: [..., 2l+1, 2k+1]
            # f[k]: [..., C, 2k+1]

            # bring channels into einsum explicitly
            # result: [..., C, 2l+1]
            out[l] = out[l] + torch.einsum("...ij,...cj->...ci", W, f[k])

    return out
  
def verify_cg_orthogonality(max_degree: int = 2, tol: float = 1e-5) -> None:
    print("Verifying CG matrix orthogonality...")
    all_ok = True
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            cg = clebsch_gordan_matrix(l, k)   # {J: [(2l+1)(2k+1), 2J+1]}
 
            # Stack all J-slices horizontally: Q^{lk} of shape [(2l+1)(2k+1), sum_J (2J+1)]
            # sum_J (2J+1) for J in |l-k|..l+k  equals  (2l+1)(2k+1) 
            Q_full = torch.cat([cg[J] for J in sorted(cg.keys())], dim=1)  # [D, D]
            D = Q_full.shape[0]
 
            # Q @ Q^T should be I_D
            QQT = Q_full @ Q_full.T
            eye = torch.eye(D, dtype=QQT.dtype)
            err = (QQT - eye).abs().max().item()
 
            status = "OK" if err < tol else "FAIL"
            if err >= tol:
                all_ok = False
            print(f"  (l={l}, k={k}): max |QQ^T - I| = {err:.2e}  [{status}]")
 
    if all_ok:
        print("All CG matrices passed orthogonality check.")
    else:
        raise ValueError(
            "One or more CG matrices failed the orthogonality check. "
            "Check sympy version and CG coefficient indexing."
        )

def direct_sum_inner_product(
    a: Dict[int, torch.Tensor],
    b: Dict[int, torch.Tensor],
) -> torch.Tensor:
    result = None
    for l in a:
        if l not in b:
            print("l not in b")
            continue
        dot = (a[l] * b[l]).sum(dim=-1)  # [...]
        result = dot if result is None else result + dot
    if result is None:
        raise ValueError("No shared degrees between query and key.")
    return result


def softmax_over_neighbors(
    scores: torch.Tensor,       # [..., num_neighbors]
    mask: torch.Tensor = None,  # [..., num_neighbors] bool, True = valid
) -> torch.Tensor:
    """
    Masked softmax along the neighbor dimension (last dim).
    """
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.nan_to_num(attn, nan=0.0)