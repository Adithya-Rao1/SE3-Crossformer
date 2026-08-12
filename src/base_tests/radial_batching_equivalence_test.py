"""
Numerical equivalence check for the vectorized apply_direct_sum_W and
_cross_update readout batching (src/se3_crossformer/se3_utils.py, model.py).

Both rewrites group per-(l,k,J) small-module forward calls (radial nets, and
EquivariantNormReadout message-gate nets) by shared input and evaluate each
group in one batched pass (stacked weights + batched matmul) instead of one
Python forward call per (l,k,J), to cut kernel-launch overhead. A bug in the
grouping/stacking (e.g. wrong ordering) could still produce an output that is
*equivariant* -- equivariance_test.py wouldn't catch it -- while being
numerically wrong. These tests compare each batched implementation against a
deliberately naive, unbatched reference that mirrors the pre-vectorization
code path exactly, on identical random inputs.
"""
import math
from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn

from src.se3_crossformer.se3_utils import (
    apply_direct_sum_W,
    clebsch_gordan_matrix,
    get_spherical_harmonics,
    EquivariantNormReadout,
    _batched_norm_readout_forward,
    RadialNetworkGRBF,
    RadialNetworkGSFB,
)
from src.se3_crossformer.irr_rep import x_to_alpha_beta

torch.manual_seed(0)

ATOL = 1e-5
RTOL = 1e-5


def _build_radial_nets(radial_net_cls, max_degree, hidden_dim, num_heads, edge_feature_dim):
    radial_nets = nn.ModuleDict()
    for l in range(max_degree + 1):
        for k in range(max_degree + 1):
            key = f"{l}_{k}"
            radial_nets[key] = nn.ModuleDict({
                str(J): radial_net_cls(
                    num_basis=(2 * l + 1), hidden_dim=hidden_dim,
                    num_heads=num_heads, edge_feature_dim=edge_feature_dim,
                )
                for J in range(abs(l - k), l + k + 1)
            })
    return radial_nets


@torch.no_grad()
def _naive_apply_direct_sum_W(
    f:           Dict[int, torch.Tensor],
    x:           torch.Tensor,
    radial_nets: nn.ModuleDict,
    max_degree:  int,
    num_heads:   int = 1,
    edge_feat:   Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Deliberately unoptimized: one radial-net forward call per (l,k,J), no
    hoisting, no memoization, no batching. Mirrors the exact pre-vectorization
    implementation of equivariant_weight_matrix + apply_direct_sum_W."""
    device, dtype = x.device, x.dtype
    N = x.shape[0]

    out: Dict[int, torch.Tensor] = {}
    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C = any_k.shape[-2]
        out[l] = torch.zeros(*x.shape[:-1], C, 2 * l + 1, device=device, dtype=dtype)

    if N == 0:
        return out

    r = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    alphas, betas = x_to_alpha_beta(x)

    for l in range(max_degree + 1):
        any_k = next(iter(f.values()))
        C = any_k.shape[-2]
        head_dim = C // num_heads

        for k in range(max_degree + 1):
            key = f"{l}_{k}"
            cg = clebsch_gordan_matrix(l, k)
            any_net = next(iter(radial_nets[key].values()))
            H = any_net.num_heads

            W = torch.zeros(N, H, 2 * l + 1, 2 * k + 1, device=device, dtype=dtype)
            for J in range(abs(l - k), l + k + 1):
                if J not in cg or str(J) not in radial_nets[key]:
                    continue
                rnet = radial_nets[key][str(J)]
                phi = rnet(r, edge_feat)
                Q_J = cg[J].to(device=device, dtype=dtype)
                Y_J = get_spherical_harmonics(J, theta=(math.pi - betas), phi=alphas)
                QTY = torch.einsum("ji,ni->nj", Q_J, Y_J)
                had = phi.unsqueeze(-1) * QTY.unsqueeze(1)
                W = W + had.view(N, H, 2 * l + 1, 2 * k + 1)

            f_k_headed = f[k].reshape(*f[k].shape[:-2], num_heads, head_dim, f[k].shape[-1])
            contrib = torch.einsum("...hij,...hcj->...hci", W, f_k_headed)
            contrib = contrib.reshape(*contrib.shape[:-3], C, 2 * l + 1)
            out[l] = out[l] + contrib

    return out


@pytest.mark.parametrize("radial_net_cls", [RadialNetworkGRBF, RadialNetworkGSFB])
@pytest.mark.parametrize("num_heads", [1, 4, 32])   # 32 == feature_dim: the depth-wise regime used in production
@pytest.mark.parametrize("bond_feature_dim", [0, 4])
def test_batched_matches_naive(radial_net_cls, num_heads, bond_feature_dim):
    max_degree  = 2
    feature_dim = 32
    hidden_dim  = 16
    N           = 10

    radial_nets = _build_radial_nets(
        radial_net_cls, max_degree, hidden_dim, num_heads, bond_feature_dim
    ).double()

    x = torch.randn(N, 3, dtype=torch.double)
    f = {
        l: torch.randn(N, feature_dim, 2 * l + 1, dtype=torch.double)
        for l in range(max_degree + 1)
    }
    edge_feat = (
        torch.randn(N, bond_feature_dim, dtype=torch.double)
        if bond_feature_dim > 0 else None
    )

    naive   = _naive_apply_direct_sum_W(
        f, x, radial_nets, max_degree, num_heads=num_heads, edge_feat=edge_feat
    )
    batched = apply_direct_sum_W(
        f, x, radial_nets, max_degree, num_heads=num_heads, edge_feat=edge_feat
    )

    for l in range(max_degree + 1):
        torch.testing.assert_close(naive[l], batched[l], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("num_members", [1, 5])
def test_batched_norm_readout_matches_naive(num_members):
    C   = 16
    dim = 5   # e.g. 2*k+1 for k=2
    S   = 12

    nets = nn.ModuleList([
        EquivariantNormReadout(C) for _ in range(num_members)
    ]).double()

    x = torch.randn(S, C, dim, dtype=torch.double)

    naive   = torch.stack([net(x) for net in nets], dim=1)   # (S, M)
    batched = _batched_norm_readout_forward(list(nets), x)   # (S, M)

    torch.testing.assert_close(naive, batched, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
