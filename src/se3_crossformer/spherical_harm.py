"""
spherical_harm.py  (optimized)
-------------------------------
Key changes vs. original
  • lpmv: replaced deep Python recursion with a fully-unrolled, tensor-native
    iterative computation.  For each (l, m) pair the entire [N] batch is
    computed in O(l) tensor ops instead of O(l) Python stack frames × many
    elementwise kernel launches per frame.

  • get_spherical_harmonics_element: now calls the iterative lpmv; no change
    to the public signature.

  • get_spherical_harmonics: the Python `for m in range(...)` loop is replaced
    by a single vectorised pass that:
      1. Builds cos(theta) once                      → [N]
      2. Computes ALL lpmv values for a given l      → [N, 2l+1]  (one sweep)
      3. Builds ALL cos/sin(m*phi) simultaneously    → [N, 2l+1]
      4. Applies normalization constants (scalars)
      5. Stacks with a single torch.stack            → [N, 2l+1]

    This collapses (2l+1) separate Python iterations + (2l+1) separate lpmv
    recursion trees into a handful of batched tensor ops and a single
    torch.stack.

  • clear_spherical_harmonics_cache: kept as a no-op stub for API compat.
"""

from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Union

import torch

# ---------------------------------------------------------------------------
# Scalar helpers (pure Python, LRU-cached — never store tensors)
# ---------------------------------------------------------------------------

CACHE = {}

def clear_spherical_harmonics_cache():
    """No-op stub kept for backward compatibility."""
    CACHE.clear()


@lru_cache(maxsize=1000)
def semifactorial(x: int) -> float:
    """x!! = x * (x-2) * (x-4) * ... (down to 1 or 2)."""
    if x <= 1:
        return 1.0
    return float(reduce(mul, range(x, 1, -2), 1))


@lru_cache(maxsize=1000)
def pochhammer(x: int, k: int) -> float:
    """Rising factorial x^(k) = x(x+1)...(x+k-1)."""
    if k <= 0:
        return 1.0
    return float(reduce(mul, range(x, x + k), 1))


# Precompute all normalization constants used in the SH formula up to l=6
# so the hot path only does float lookups, not repeated math.
@lru_cache(maxsize=512)
def _sh_norm(l: int, m: int) -> float:
    """
    Normalization factor for the real tesseral harmonic Y_l^m.
    For m == 0:   sqrt((2l+1) / (4*pi))
    For m != 0:   sqrt((2l+1) / (4*pi)) * sqrt(2 / pochhammer(l-|m|+1, 2|m|))
    """
    m_abs = abs(m)
    N = sqrt((2 * l + 1) / (4 * pi))
    if m_abs == 0:
        return N
    return N * sqrt(2.0 / pochhammer(l - m_abs + 1, 2 * m_abs))


@lru_cache(maxsize=512)
def _cs_phase_semifact(m_abs: int) -> float:
    """(-1)^m * (2m-1)!!  — the seed of the P_l^l recursion."""
    return float((-1) ** m_abs * semifactorial(2 * m_abs - 1))


# ---------------------------------------------------------------------------
# Iterative Associated Legendre polynomials  [N] → [N]
# ---------------------------------------------------------------------------

def lpmv(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """
    Associated Legendre polynomial P_l^m(x) including Condon-Shortley phase.

    Replaces the original recursive implementation.  The standard three-term
    recurrence is unrolled into an explicit Python loop whose *body* is a
    handful of elementwise tensor ops.  The Python loop runs at most `l` times
    regardless of batch size N; the GPU sees O(l) kernel launches instead of
    the original O(l^2) launches from nested recursion.

    Args:
        l : int  — degree
        m : int  — order (may be negative)
        x : Tensor[N]  — argument (typically cos θ)
    Returns:
        Tensor[N]
    """
    m_abs = abs(m)

    if m_abs > l:
        return torch.zeros_like(x)

    # --- seed: P_m^m ---
    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    pmm: torch.Tensor
    if m_abs == 0:
        pmm = torch.ones_like(x)
    else:
        sin_theta_sq = (1.0 - x * x).clamp(min=0.0)        # [N], numerical guard
        pmm = _cs_phase_semifact(m_abs) * sin_theta_sq.pow(m_abs * 0.5)

    if l == m_abs:
        y = pmm
        if m < 0:
            # P_l^{-|m|} = (-1)^|m| * (l-|m|)! / (l+|m|)! * P_l^{|m|}
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # --- one-step: P_{m+1}^m ---
    pmmp1 = x * (2 * m_abs + 1) * pmm                       # [N]

    if l == m_abs + 1:
        y = pmmp1
        if m < 0:
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # --- standard three-term recurrence up to degree l ---
    p_prev2 = pmm     # P_{ll}^m
    p_prev1 = pmmp1   # P_{l+1,l}^m  (i.e. P_{m+1}^m at start)
    p_curr  = pmmp1   # will be overwritten immediately

    for ll in range(m_abs + 2, l + 1):
        # (ll - m_abs) * P_ll^m = (2*ll-1)*x*P_{ll-1}^m - (ll+m_abs-1)*P_{ll-2}^m
        p_curr  = ((2 * ll - 1) * x * p_prev1 - (ll + m_abs - 1) * p_prev2) / (ll - m_abs)
        p_prev2 = p_prev1
        p_prev1 = p_curr

    y = p_curr
    if m < 0:
        y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
    return y


# ---------------------------------------------------------------------------
# Vectorised spherical harmonics
# ---------------------------------------------------------------------------

def get_spherical_harmonics_element(
    l: int,
    m: int,
    theta: torch.Tensor,   # [N]
    phi:   torch.Tensor,   # [N]
) -> torch.Tensor:          # [N]
    """
    Single (l, m) tesseral spherical harmonic.
    Signature unchanged from original; now calls the iterative lpmv.
    """
    m_abs = abs(m)
    assert m_abs <= l

    norm = _sh_norm(l, m)                              # scalar
    leg  = lpmv(l, m_abs, torch.cos(theta))            # [N]

    if m == 0:
        return norm * leg

    trig = torch.cos(m * phi) if m > 0 else torch.sin(m_abs * phi)
    return norm * leg * trig


def get_spherical_harmonics(
    l:     int,
    theta: torch.Tensor,   # [N] or scalar
    phi:   torch.Tensor,   # [N] or scalar
) -> torch.Tensor:          # [N, 2l+1]  (or [2l+1] for scalar input)
    """
    All 2l+1 tesseral spherical harmonics for degree l.

    Optimized vs. original
    ──────────────────────
    Original: Python loop over m=−l..l, each iteration calling lpmv (which was
    itself recursive).  Total kernel launches ≈ (2l+1) × O(l²).

    New: single vectorised pass —
      1. cos(theta) computed once                       → [N]
      2. lpmv called once per unique |m|  (l+1 calls)  → [N] each
         (no redundant work: P_l^{|m|} is reused for +m and −m)
      3. All trig values built as [N, 2l+1] in two einsum-free ops
      4. One torch.stack at the end

    Total kernel launches ≈ O(l) tensor ops + 1 stack.
    """
    scalar_input = theta.dim() == 0
    if scalar_input:
        theta = theta.unsqueeze(0)
        phi   = phi.unsqueeze(0)

    cos_theta = torch.cos(theta)                       # [N]

    # Pre-compute lpmv for all unique |m| values in one pass
    # unique |m| values: 0, 1, ..., l  (total l+1 calls)
    lpmv_cache: dict[int, torch.Tensor] = {}
    for m_abs in range(l + 1):
        lpmv_cache[m_abs] = lpmv(l, m_abs, cos_theta) # [N]

    # Build trig [N, 2l+1]:
    # m_vals = [-l, ..., -1, 0, 1, ..., l]
    m_vals = torch.arange(-l, l + 1, device=theta.device, dtype=theta.dtype)   # [2l+1]
    m_abs_vals = m_vals.abs()                                                    # [2l+1]

    # cos(m*phi) for m>0 and sin(|m|*phi) for m<0 simultaneously:
    # Build angle matrix: [N, 2l+1]
    phi_exp = phi.unsqueeze(1) * m_abs_vals.unsqueeze(0)   # [N, 2l+1]

    # For m>0: cos(|m|*phi),  m==0: 1,  m<0: sin(|m|*phi)
    trig = torch.where(
        m_vals.unsqueeze(0) > 0,
        torch.cos(phi_exp),
        torch.where(
            m_vals.unsqueeze(0) < 0,
            torch.sin(phi_exp),
            torch.ones_like(phi_exp),
        ),
    )   # [N, 2l+1]

    # Stack lpmv values into [N, 2l+1], reusing cache
    leg_stack = torch.stack(
        [lpmv_cache[int(m_abs_vals[i].item())] for i in range(2 * l + 1)],
        dim=1,
    )   # [N, 2l+1]

    # Normalization constants as [2l+1] tensor
    norms = torch.tensor(
        [_sh_norm(l, int(m)) for m in range(-l, l + 1)],
        dtype=theta.dtype, device=theta.device,
    )   # [2l+1]

    Y = norms.unsqueeze(0) * leg_stack * trig   # [N, 2l+1]

    if scalar_input:
        return Y.squeeze(0)
    return Y