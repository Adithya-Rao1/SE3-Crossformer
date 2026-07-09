"""
irr_rep.py  (patched)
---------------------
Key changes vs. original
  • x_to_alpha_beta: fully vectorised — no Python for-loop over atoms.
    Returns (Tensor[N], Tensor[N]) instead of (list, list).
  • All callers in se3_utils.equivariant_weight_matrix are updated to
    consume the tensor pair directly.
"""

import os
import numpy as np
import torch
from torch import sin, cos, atan2, acos
from math import pi
from pathlib import Path
from functools import wraps

from src.se3_crossformer.utils import exists, default, cast_torch_tensor, to_order
from src.se3_crossformer.spherical_harm import get_spherical_harmonics, clear_spherical_harmonics_cache

DATA_PATH = Path('/home/ubuntu/se3-crossformer-data/se3-transformer-pytorch/se3_transformer_pytorch/data')

try:
    path = DATA_PATH / 'J_dense.pt'
    Jd = torch.load(str(path))
except Exception:
    path = DATA_PATH / 'J_dense.npy'
    Jd_np = np.load(str(path), allow_pickle=True)
    Jd = list(map(torch.from_numpy, Jd_np))


def wigner_d_matrix(degree, alpha, beta, gamma, dtype=None, device=None):
    """Create wigner D matrices for batch of ZYZ Euler angles for degree l."""
    J = Jd[degree].type(dtype).to(device)
    x_a = z_rot_mat(alpha, degree)
    x_b = z_rot_mat(beta,  degree)
    x_c = z_rot_mat(gamma, degree)
    res = x_a @ J @ x_b @ J @ x_c
    order = to_order(degree)
    return res.view(order, order)


def z_rot_mat(angle, l):
    device, dtype = angle.device, angle.dtype
    order = to_order(l)
    m = angle.new_zeros((order, order))
    inds          = torch.arange(0, order, 1,      dtype=torch.long,  device=device)
    reversed_inds = torch.arange(2 * l, -1, -1,   dtype=torch.long,  device=device)
    frequencies   = torch.arange(l, -l - 1, -1,   dtype=dtype,       device=device)[None]
    m[inds, reversed_inds] = sin(frequencies * angle[None])
    m[inds, inds]          = cos(frequencies * angle[None])
    return m


def irr_repr(order, alpha, beta, gamma, dtype=None):
    """Irreducible representation of SO3."""
    cast_ = cast_torch_tensor(lambda t: t)
    dtype = default(dtype, torch.get_default_dtype())
    alpha, beta, gamma = map(cast_, (alpha, beta, gamma))
    return wigner_d_matrix(order, alpha, beta, gamma, dtype=dtype)


@cast_torch_tensor
def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma),  cos(gamma), 0],
        [0,           0,          1],
    ], dtype=gamma.dtype)


@cast_torch_tensor
def rot_y(beta):
    return torch.tensor([
        [ cos(beta), 0, sin(beta)],
        [0,          1, 0        ],
        [-sin(beta), 0, cos(beta)],
    ], dtype=beta.dtype)


# ── vectorised x_to_alpha_beta ────────────────────────────────────────────────

def x_to_alpha_beta(x: torch.Tensor):
    """
    Convert Cartesian direction(s) on the unit sphere to (alpha, beta).

    Supports:
      • 1-D input  [3]        → returns (scalar, scalar)  [unchanged API]
      • 2-D input  [N, 3]     → returns (Tensor[N], Tensor[N])  VECTORISED
                                 (previously used a Python for-loop)

    The normalisation and clamping are done with tensor ops so the whole
    batch runs as a single CUDA kernel launch rather than N sequential ones.
    """
    if x.ndim == 1:
        # ── scalar path (unchanged) ───────────────────────────────────────
        x = x / (x.norm() + 1e-8)
        beta  = acos(x[2].clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        alpha = atan2(x[1], x[0])
        return alpha, beta

    # ── batched path [N, 3] ───────────────────────────────────────────────
    # Normalise all rows in one shot
    norms  = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [N, 1]
    x_norm = x / norms                                        # [N, 3]

    # Clamp z to the valid range of acos to avoid NaN at ±1
    z_safe = x_norm[:, 2].clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    alphas = torch.atan2(x_norm[:, 1], x_norm[:, 0])   # [N]
    betas  = torch.acos(z_safe)                          # [N]

    return alphas, betas                                  # both Tensor[N]


def rot(alpha, beta, gamma):
    """ZYZ Euler angles rotation."""
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


def compose(a1, b1, c1, a2, b2, c2):
    comp = rot(a1, b1, c1) @ rot(a2, b2, c2)
    xyz  = comp @ torch.tensor([0, 0, 1.])
    a, b = x_to_alpha_beta(xyz)
    rotz = rot(0, -b, -a) @ comp
    c    = atan2(rotz[1, 0], rotz[0, 0])
    return a, b, c


def spherical_harmonics(order, alpha, beta, dtype=None):
    return get_spherical_harmonics(order, theta=(pi - beta), phi=alpha)