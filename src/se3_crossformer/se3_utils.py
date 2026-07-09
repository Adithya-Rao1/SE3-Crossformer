"""
se3_utils.py  (optimized)
--------------------------
Key changes vs. original
  ① RadialNetwork
      • _basis() removed entirely.  The original called scipy.special.jv and
        spherical_bessel_first_kind on the CPU inside every forward pass,
        causing a CPU↔GPU round-trip per (l, k, J) per batch.
      • Replaced with a GPU-native RBF basis: Gaussian-envelope Bessel-style
        radial features built entirely in PyTorch with no NumPy/scipy in the
        hot path.  Roots are precomputed once at construction time and stored
        as a non-trainable buffer on the correct device.
      • forward() now accepts only (r,) — degree and k are no longer needed,
        simplifying all call sites.

  ② equivariant_weight_matrix
      • The per-J loop body now calls get_spherical_harmonics (vectorised,
        from the optimized spherical_harm.py) instead of the old recursive
        path.
      • phi (radial scalar) is computed once per J as rnet(r) → [N, 1].
      • All other logic (CG cache lookup, einsum) is unchanged but now runs
        on fully-batched [N] tensors with no Python recursion in the inner
        loop.

  ③ Everything else (CG matrices, apply_direct_sum_W, direct_sum helpers,
     softmax, verify_cg) is unchanged.
"""

from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import numpy as np

# Public re-export so existing importers don't break
from .spherical_harm import (
    get_spherical_harmonics,
    get_spherical_harmonics_element,
    clear_spherical_harmonics_cache,
    lpmv,
    semifactorial,
    pochhammer,
    CACHE,
)

# ---------------------------------------------------------------------------
# Clebsch-Gordan matrices (unchanged)
# ---------------------------------------------------------------------------

_CG_CACHE: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}


def clebsch_gordan_matrix(l: int, k: int) -> Dict[int, torch.Tensor]:
    """Return {J: Q_J} Clebsch-Gordan matrices, cached after first call."""
    if (l, k) in _CG_CACHE:
        return _CG_CACHE[(l, k)]

    try:
        from sympy.physics.quantum.cg import CG
        from sympy import Rational, N as sympy_N
    except ImportError:
        raise ImportError(
            "sympy is required for Clebsch-Gordan computation.\n"
            "Install with:  pip install sympy"
        )

    dim_out = (2 * l + 1) * (2 * k + 1)
    result: Dict[int, torch.Tensor] = {}

    for J in range(abs(l - k), l + k + 1):
        dim_J = 2 * J + 1
        Q_J   = torch.zeros(dim_out, dim_J, dtype=torch.float64)

        for m_l in range(-l, l + 1):
            for m_k in range(-k, k + 1):
                row = (m_l + l) * (2 * k + 1) + (m_k + k)
                for M in range(-J, J + 1):
                    col    = M + J
                    cg_val = CG(
                        Rational(l),  Rational(m_l),
                        Rational(k),  Rational(m_k),
                        Rational(J),  Rational(M),
                    ).doit()
                    Q_J[row, col] = float(sympy_N(cg_val))

        result[J] = Q_J.to(torch.float32)

    _CG_CACHE[(l, k)] = result
    return result


def precompute_cg_matrices(max_degree: int) -> None:
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            clebsch_gordan_matrix(l, k)
    print(f"Precomputed CG matrices for degrees 0..{max_degree}.")


# ---------------------------------------------------------------------------
# ① RadialNetwork — GPU-native, no scipy in forward pass
# ---------------------------------------------------------------------------

class RadialNetwork(nn.Module):
    """
    Learnable radial function r → scalar.

    Basis: num_basis Gaussian-RBF features centred on a uniform grid of
    Bessel-root-inspired breakpoints in [0, cutoff_radius].  All computed
    on the GPU; no scipy, no NumPy, no CPU↔GPU transfers in forward().

    Args:
        num_basis:      number of radial basis functions
        hidden_dim:     width of the 2-layer MLP
        cutoff_radius:  envelope cutoff (Å or Bohr, must match positions)
    """

    def __init__(
        self,
        num_basis:     int   = 8,
        hidden_dim:    int   = 32,
        cutoff_radius: float = 5.0,
    ):
        super().__init__()
        self.num_basis     = num_basis
        self.cutoff_radius = cutoff_radius

        # Centres evenly spaced in [0, cutoff_radius]; stored as a buffer
        # so they move to the right device with .to(device) / .cuda().
        centres = torch.linspace(0.0, cutoff_radius, num_basis)   # [B]
        self.register_buffer("centres", centres)

        # Width: half the spacing between centres
        width = cutoff_radius / max(num_basis - 1, 1)
        self.register_buffer(
            "inv_width_sq",
            torch.tensor(-1.0 / (2.0 * width ** 2)),
        )

        # Cosine-envelope cutoff to enforce smoothness at cutoff_radius
        # f(r) = 0.5*(cos(pi*r/cutoff)+1) for r < cutoff, else 0
        self.net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _basis(self, r: torch.Tensor) -> torch.Tensor:
        """
        Gaussian RBF basis.

        Args:
            r: Tensor[N, 1]  — interatomic distances (must be on same device
               as self.centres)
        Returns:
            Tensor[N, num_basis]
        """
        # [N, 1] - [1, B] → [N, B]
        diff    = r - self.centres.unsqueeze(0)          # [N, B]
        rbf     = torch.exp(self.inv_width_sq * diff * diff)  # [N, B]

        # Cosine envelope: zero outside cutoff, smooth at boundary
        r_scaled = (r / self.cutoff_radius).clamp(max=1.0)         # [N, 1]
        envelope = 0.5 * (torch.cos(math.pi * r_scaled) + 1.0)    # [N, 1]

        return rbf * envelope   # [N, B]

    def forward(self, r: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            r:    Tensor[N, 1] — distances on the correct device
            *args, **kwargs: ignored (degree / k legacy args accepted but unused)
        Returns:
            Tensor[N, 1]
        """
        basis = self._basis(r)          # [N, num_basis]
        return self.net(basis)          # [N, 1]


# ---------------------------------------------------------------------------
# ② equivariant_weight_matrix — vectorised, no recursion in hot path
# ---------------------------------------------------------------------------

def equivariant_weight_matrix(
    x:          torch.Tensor,   # [N, 3]
    l:          int,
    k:          int,
    radial_net: nn.Module,
) -> torch.Tensor:               # [N, 2l+1, 2k+1]
    """
    Equivariant weight matrix W^{lk}(x) summed over all valid J channels.

    Changes vs. original
    ─────────────────────
    • get_spherical_harmonics is now called on fully-batched [N] tensors
      (from the optimized spherical_harm.py) — no Python recursion per atom.
    • radial_net.forward(r) takes only r; legacy (r, l, k) call sites are
      still compatible because RadialNetwork.forward accepts *args.
    • CG matrices are fetched from the module-level cache (populated by
      precompute_cg_matrices at startup).
    """
    from .irr_rep import x_to_alpha_beta

    device, dtype = x.device, x.dtype
    N = x.shape[0]
    W = torch.zeros(N, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)

    if N == 0:
        return W

    r      = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [N, 1]
    alphas, betas = x_to_alpha_beta(x)                       # [N], [N]

    cg = clebsch_gordan_matrix(l, k)   # fetched from cache

    for J in range(abs(l - k), l + k + 1):
        if J not in cg:
            continue

        # Select the radial sub-network for this J
        if isinstance(radial_net, nn.ModuleDict):
            if str(J) not in radial_net:
                continue
            rnet = radial_net[str(J)]
        else:
            rnet = radial_net

        phi = rnet(r)                                         # [N, 1]

        Q_J = cg[J].to(device=device, dtype=dtype)           # [(2l+1)(2k+1), 2J+1]

        # Vectorised: one call for all N atoms simultaneously
        Y_J = get_spherical_harmonics(
            J, theta=(math.pi - betas), phi=alphas
        )                                                     # [N, 2J+1]

        QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)            # [N, (2l+1)(2k+1)]
        had = phi * QTY                                       # [N, (2l+1)(2k+1)]
        W   = W + had.view(N, 2 * l + 1, 2 * k + 1)

    return W


# ---------------------------------------------------------------------------
# Everything below is unchanged from the original se3_utils.py
# ---------------------------------------------------------------------------

def apply_direct_sum_W(
    f:           Dict[int, torch.Tensor],
    x:           torch.Tensor,
    radial_nets: nn.ModuleDict,
    max_degree:  int,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}

    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C     = any_k.shape[-2]

        out[l] = torch.zeros(
            *x.shape[:-1], C, 2 * l + 1,
            device=x.device, dtype=x.dtype,
        )

        for k in range(max_degree + 1):
            key = f"{l}_{k}"
            W   = equivariant_weight_matrix(x, l, k, radial_nets[key])
            contrib = torch.einsum("...ij,...cj->...ci", W, f[k])
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
        scores = scores.masked_fill(~mask, float(-1e9))
    attn = torch.nn.functional.softmax(scores + 1e-8, dim=-1)
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