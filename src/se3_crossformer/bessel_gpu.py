import numpy as np
import time
from scipy.special import spherical_jn as _scipy_jn
import torch


def _spherical_jn_table(ell_values, x_nodes_per_seg, n_seg, N_cheb):
    import warnings
    N_ell = len(ell_values)
    ell_max = int(torch.as_tensor(ell_values).max())

    jl_table = torch.zeros((N_ell, n_seg, N_cheb))
    jlp_table = torch.zeros((N_ell, n_seg, N_cheb))

    seg_x_min = torch.min(x_nodes_per_seg, dim=1)
    seg_x_max = torch.max(x_nodes_per_seg, dim=1)

    all_x = x_nodes_per_seg.ravel().clamp(min=1e-30)
    inv_x = 1.0 / all_x

    j_prev = torch.sin(all_x) * inv_x
    j_prev[all_x < 1e-30] = 1.0
    j_curr = (torch.sin(all_x) * inv_x - torch.cos(all_x)) * inv_x
    j_curr[all_x < 1e-30] = 0.0

    ell_to_idx = {int(e): i for i, e in enumerate(ell_values)}

    if 0 in ell_to_idx:
        jl_table[ell_to_idx[0]] = j_prev.reshape(n_seg, N_cheb)
        jlp_table[ell_to_idx[0]] = (-j_curr).reshape(n_seg, N_cheb)
    if 1 in ell_to_idx:
        jl_table[ell_to_idx[1]] = j_curr.reshape(n_seg, N_cheb)
        jlp_table[ell_to_idx[1]] = (j_prev - 2.0 * inv_x * j_curr).reshape(n_seg, N_cheb)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ell in range(1, ell_max):
            j_next = (2 * ell + 1) * inv_x * j_curr - j_prev
            l_new = ell + 1

            if l_new in ell_to_idx:
                il = ell_to_idx[l_new]
                jl_table[il] = j_next.reshape(n_seg, N_cheb)
                jlp_table[il] = (j_curr - (l_new + 1) * inv_x * j_next).reshape(n_seg, N_cheb)

            j_prev = j_curr
            j_curr = j_next

    for il, ell in enumerate(ell_values):
        ell_int = int(ell)
        if ell_int <= 1:
            continue

        needs_fix = []
        for s in range(n_seg):
            if seg_x_min[s] > 1.5 * ell_int:
                continue
            elif seg_x_max[s] < 0.3 * ell_int:
                jl_table[il, s, :] = 0.0
                jlp_table[il, s, :] = 0.0
            else:
                needs_fix.append(s)

        if not needs_fix:
            continue

        n_fix = len(needs_fix)
        x_batch = torch.empty(n_fix * N_cheb)
        for i, s in enumerate(needs_fix):
            x_batch[i * N_cheb:(i + 1) * N_cheb] = x_nodes_per_seg[s]

        jl_batch = _scipy_jn(ell_int, x_batch)
        jlp_batch = _scipy_jn(ell_int, x_batch, derivative=True)

        for i, s in enumerate(needs_fix):
            jl_table[il, s] = torch.from_numpy(jl_batch[i * N_cheb:(i + 1) * N_cheb]).float()
            jlp_table[il, s] = torch.from_numpy(jlp_batch[i * N_cheb:(i + 1) * N_cheb]).float()

    bad = ~torch.isfinite(jl_table)
    jl_table[bad] = 0.0
    bad_d = ~torch.isfinite(jlp_table)
    jlp_table[bad_d] = 0.0

    return jl_table, jlp_table

class BesselTable:
    def __init__(self, ell_values, x_max=5500.0, N_cheb=64,
                 seg_width=80.0, verbose=True):
        self.ell_values = torch.tensor(ell_values, dtype=torch.int64)
        self.N_ell = len(self.ell_values)
        self.x_max = float(x_max)
        self.N_cheb = int(N_cheb)

        t0 = time.time()
        self._build_table(seg_width)
        dt = time.time() - t0

        if verbose:
            print(f"[BesselTable] Built: {self.N_ell} ells, "
                  f"{self.n_seg} segs x {self.N_cheb} nodes, "
                  f"x_max={self.x_max:.0f}, "
                  f"time={dt:.2f}s")

    def _build_table(self, seg_width):
        N_cheb = self.N_cheb
        x_max = self.x_max

        n_seg = max(1, int(np.ceil(x_max / seg_width)))
        seg_edges = torch.linspace(0.0, x_max, n_seg + 1)
        self.n_seg = n_seg
        self.seg_width_actual = x_max / n_seg

        t_nodes = torch.cos(torch.pi * (torch.arange(N_cheb) + 0.5) / N_cheb)

        x_nodes = torch.empty((n_seg, N_cheb))
        for s in range(n_seg):
            a, b = seg_edges[s], seg_edges[s + 1]
            x_nodes[s] = 0.5 * (b - a) * t_nodes + 0.5 * (a + b)
        x_nodes = x_nodes.clamp(min=1e-15)

        jl_segs, jlp_segs = _spherical_jn_table(
            self.ell_values, x_nodes, n_seg, N_cheb)

        k_idx = torch.arange(N_cheb)
        arccos_t = torch.arccos(t_nodes)
        DCT_mat = torch.cos(k_idx[:, None] * arccos_t[None, :])
        DCT_mat *= (2.0 / N_cheb)
        DCT_mat[0] *= 0.5

        self.coeffs = torch.einsum('ijk,lk->ijl', jl_segs, DCT_mat)
        self.coeffs_d = torch.einsum('ijk,lk->ijl', jlp_segs, DCT_mat)

    def _map_to_segments(self, x_arr_mx):
        sw = self.seg_width_actual
        x = torch.clip(x_arr_mx, 0.0, self.x_max - 1e-6)

        seg_idx = torch.floor(x / sw)
        seg_idx = torch.clip(seg_idx, 0, self.n_seg - 1)

        a = seg_idx.float() * sw
        t_local = 2.0 * (x - a) / sw - 1.0

        return seg_idx, t_local

    def _eval_chebyshev(self, coeffs, seg_idx, t_local):
        N_cheb = self.N_cheb
        c = coeffs[:, seg_idx.long(), :]
        theta = torch.arccos(torch.clip(t_local, -1.0, 1.0))
        k_idx = torch.arange(N_cheb).float()
        T_basis = torch.cos(theta[:, None] * k_idx[None, :])
        result = torch.sum(c * T_basis[None, :, :], dim=2)
        return result

    def eval_jl(self, x_arr):
        x_mx = torch.as_tensor(x_arr)
        seg_idx, t = self._map_to_segments(x_mx)
        return self._eval_chebyshev(self.coeffs, seg_idx, t)

    def eval_jlp(self, x_arr):
        x_mx = torch.as_tensor(x_arr)
        seg_idx, t = self._map_to_segments(x_mx)
        return self._eval_chebyshev(self.coeffs_d, seg_idx, t)

    def eval_jl_jlp(self, x_arr):
        x_mx = torch.as_tensor(x_arr)
        seg_idx, t = self._map_to_segments(x_mx)
        jl = self._eval_chebyshev(self.coeffs, seg_idx, t)
        jlp = self._eval_chebyshev(self.coeffs_d, seg_idx, t)
        return jl, jlp