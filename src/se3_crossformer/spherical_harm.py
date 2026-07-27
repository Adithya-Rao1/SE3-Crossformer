from math import pi, sqrt
from functools import reduce, lru_cache
from operator import mul
from typing import Union

import torch

CACHE = {}

def clear_spherical_harmonics_cache():
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

def lpmv(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """
    Associated Legendre polynomial P_l^m(x) including Condon-Shortley phase.
    """
    m_abs = abs(m)

    if m_abs > l:
        return torch.zeros_like(x)

    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    pmm: torch.Tensor
    if m_abs == 0:
        pmm = torch.ones_like(x)
    else:
        sin_theta_sq = (1.0 - x * x).clamp(min=1e-8)       
        pmm = _cs_phase_semifact(m_abs) * sin_theta_sq.pow(m_abs * 0.5)

    if l == m_abs:
        y = pmm
        if m < 0:
            # P_l^{-|m|} = (-1)^|m| * (l-|m|)! / (l+|m|)! * P_l^{|m|}
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # one-step: P_{m+1}^m
    pmmp1 = x * (2 * m_abs + 1) * pmm                    

    if l == m_abs + 1:
        y = pmmp1
        if m < 0:
            y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
        return y

    # standard three-term recurrence up to degree l
    p_prev2 = pmm     # P_{ll}^m
    p_prev1 = pmmp1   # P_{l+1,l}^m 

    for ll in range(m_abs + 2, l + 1):
        # (ll - m_abs) * P_ll^m = (2*ll-1)*x*P_{ll-1}^m - (ll+m_abs-1)*P_{ll-2}^m
        p_curr  = ((2 * ll - 1) * x * p_prev1 - (ll + m_abs - 1) * p_prev2) / (ll - m_abs)
        p_prev2 = p_prev1
        p_prev1 = p_curr

    y = p_curr
    if m < 0:
        y = y * ((-1) ** m_abs / pochhammer(l - m_abs + 1, 2 * m_abs))
    return y

def get_spherical_harmonics_element(
    l: int,
    m: int,
    theta: torch.Tensor,   
    phi:   torch.Tensor,   
) -> torch.Tensor:         
    """
    Single (l, m) tesseral spherical harmonic.
    """
    m_abs = abs(m)
    assert m_abs <= l

    norm = _sh_norm(l, m)                              
    leg  = lpmv(l, m_abs, torch.cos(theta))            

    if m == 0:
        return norm * leg

    trig = torch.cos(m * phi) if m > 0 else torch.sin(m_abs * phi)
    return norm * leg * trig

def get_spherical_harmonics(
    l:     int,
    theta: torch.Tensor,   
    phi:   torch.Tensor, 
) -> torch.Tensor:         
    """
    All 2l+1 tesseral spherical harmonics for degree l.
    """
    scalar_input = theta.dim() == 0
    if scalar_input:
        theta = theta.unsqueeze(0)
        phi   = phi.unsqueeze(0)

    cos_theta = torch.cos(theta)                     

    lpmv_cache: dict[int, torch.Tensor] = {}
    for m_abs in range(l + 1):
        lpmv_cache[m_abs] = lpmv(l, m_abs, cos_theta) # [N]

    m_vals = torch.arange(-l, l + 1, device=theta.device, dtype=theta.dtype)  
    m_abs_vals = m_vals.abs()                                                 

    # cos(m*phi) for m>0 and sin(|m|*phi) for m<0 simultaneously
    phi_exp = phi.unsqueeze(1) * m_abs_vals.unsqueeze(0)   
    trig = torch.where(
        m_vals.unsqueeze(0) > 0,
        torch.cos(phi_exp),
        torch.where(
            m_vals.unsqueeze(0) < 0,
            torch.sin(phi_exp),
            torch.ones_like(phi_exp),
        ),
    )   

    leg_stack = torch.stack(
        [lpmv_cache[int(m_abs_vals[i].item())] for i in range(2 * l + 1)],
        dim=1,
    )  

    norms = torch.tensor(
        [_sh_norm(l, int(m)) for m in range(-l, l + 1)],
        dtype=theta.dtype, device=theta.device,
    )   

    Y = norms.unsqueeze(0) * leg_stack * trig  

    if scalar_input:
        return Y.squeeze(0)
    return Y