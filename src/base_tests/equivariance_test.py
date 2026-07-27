import math
import pytest
import torch

from src.se3_crossformer.model import (
    SE3InterNeighborhoodTransformer,
    SE3IntraOnlyTransformer,
)
from src.se3_crossformer.se3_utils import RadialNetworkGSFB

torch.manual_seed(0)

ATOL = 1e-9
RTOL = 1e-9

def random_rotation_matrix(dtype=torch.double, generator=None) -> torch.Tensor:
    """Uniformly-ish sampled proper rotation matrix (det = +1) via QR."""
    A = torch.randn(3, 3, dtype=torch.double, generator=generator)
    Q, R = torch.linalg.qr(A)
    # Fix sign ambiguity of QR so Q is Haar-ish distributed.
    d = torch.diagonal(R).sign()
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)

def qvec_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert the 5-component traceless symmetric quadrupole representation
    [Qxx, Qxy, Qxz, Qyy, Qyz] (matches _cartesian_quadrupole in model.py)
    into a full [..., 3, 3] symmetric traceless matrix.
    """
    Qxx, Qxy, Qxz, Qyy, Qyz = q.unbind(-1)
    Qzz = -Qxx - Qyy
    row0 = torch.stack([Qxx, Qxy, Qxz], dim=-1)
    row1 = torch.stack([Qxy, Qyy, Qyz], dim=-1)
    row2 = torch.stack([Qxz, Qyz, Qzz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

def make_toy_graph(n_atoms=12, n_atom_types=5, num_parts=4, seed=0):
    g = torch.Generator().manual_seed(seed)

    x = torch.randn(n_atoms, 3, generator=g).double()

    atom_type_idx = torch.randint(0, n_atom_types, (n_atoms,), generator=g)
    node_features = torch.nn.functional.one_hot(
        atom_type_idx, num_classes=n_atom_types
    ).double()

    # Ring topology as a base (guarantees connectivity), plus a few chords.
    ring_src = torch.arange(n_atoms)
    ring_dst = (ring_src + 1) % n_atoms
    chord_src = torch.randint(0, n_atoms, (n_atoms // 2,), generator=g)
    chord_dst = torch.randint(0, n_atoms, (n_atoms // 2,), generator=g)
    keep = chord_src != chord_dst
    chord_src, chord_dst = chord_src[keep], chord_dst[keep]

    src = torch.cat([ring_src, ring_dst, chord_src, chord_dst])
    dst = torch.cat([ring_dst, ring_src, chord_dst, chord_src])
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr = torch.randint(1, 4, (edge_index.shape[1], 1), generator=g).double()

    atomic_masses = torch.empty(n_atoms, dtype=torch.double).uniform_(1.0, 16.0, generator=g)

    batch = torch.zeros(n_atoms, dtype=torch.long)

    return node_features, x, edge_index, edge_attr, atomic_masses, batch

MODEL_CLASSES = [SE3InterNeighborhoodTransformer, SE3IntraOnlyTransformer]
TASKS = [0, 1, 2, 3]

def build_model(model_cls, in_features, task, num_parts=4, max_degree=2):
    model = model_cls(
        radial_net=RadialNetworkGSFB,
        in_features=in_features,
        max_degree=max_degree,
        num_layers=2,
        feature_dim=16,
        hidden_dim=32,
        num_parts=num_parts,
        scalar_out_dim=1,
        task=task,
        partition_type="spectral",
    )
    model.double().eval()
    return model

def run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch):
    with torch.no_grad():
        return model(
            node_features=node_features,
            x=x,
            edge_index=edge_index,
            atomic_masses=atomic_masses,
            batch=batch,
            edge_attr=edge_attr,
        )

def assert_scalar_invariant(out_raw, out_tf):
    torch.testing.assert_close(out_raw, out_tf, atol=ATOL, rtol=RTOL)

def assert_vector_rotates(out_raw, out_tf, R):
    expected = torch.einsum("ij,bj->bi", R, out_raw)
    torch.testing.assert_close(out_tf, expected, atol=ATOL, rtol=RTOL)

def assert_vector_invariant(out_raw, out_tf):
    torch.testing.assert_close(out_raw, out_tf, atol=ATOL, rtol=RTOL)

def assert_tensor_rotates(out_raw, out_tf, R):
    Q_raw = qvec_to_matrix(out_raw)
    Q_tf = qvec_to_matrix(out_tf)
    expected = torch.einsum("ij,bjk,lk->bil", R, Q_raw, R)
    torch.testing.assert_close(Q_tf, expected, atol=ATOL, rtol=RTOL)

def assert_tensor_invariant(out_raw, out_tf):
    torch.testing.assert_close(out_raw, out_tf, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
@pytest.mark.parametrize("task", TASKS)
def test_rotation_equivariance(model_cls, task):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=42
    )
    model = build_model(model_cls, in_features=n_atom_types, task=task)

    R = random_rotation_matrix()
    x_rot = x @ R.T 

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_rot, edge_index, edge_attr, atomic_masses, batch)

    if task == 0:
        assert_scalar_invariant(out_raw, out_tf)
    elif task == 1:
        assert_vector_rotates(out_raw, out_tf, R)
    elif task == 2:
        assert_tensor_rotates(out_raw, out_tf, R)
    else:  # task == 3 -> (scalar, vector, tensor)
        s_raw, v_raw, t_raw = out_raw
        s_tf, v_tf, t_tf = out_tf
        assert_scalar_invariant(s_raw, s_tf)
        assert_vector_rotates(v_raw, v_tf, R)
        assert_tensor_rotates(t_raw, t_tf, R)

@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
@pytest.mark.parametrize("task", TASKS)
def test_translation_invariance(model_cls, task):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=7
    )
    model = build_model(model_cls, in_features=n_atom_types, task=task)

    t = torch.tensor([3.7, -2.1, 5.5], dtype=torch.double)
    x_shifted = x + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_shifted, edge_index, edge_attr, atomic_masses, batch)

    if task == 0:
        assert_scalar_invariant(out_raw, out_tf)
    elif task == 1:
        assert_vector_invariant(out_raw, out_tf)
    elif task == 2:
        assert_tensor_invariant(out_raw, out_tf)
    else:
        s_raw, v_raw, t_raw = out_raw
        s_tf, v_tf, t_tf = out_tf
        assert_scalar_invariant(s_raw, s_tf)
        assert_vector_invariant(v_raw, v_tf)
        assert_tensor_invariant(t_raw, t_tf)

@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
@pytest.mark.parametrize("task", TASKS)
def test_combined_rotation_and_translation(model_cls, task):
    """SE(3) = rotation followed by translation, applied together."""
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=14, n_atom_types=n_atom_types, num_parts=4, seed=99
    )
    model = build_model(model_cls, in_features=n_atom_types, task=task)

    R = random_rotation_matrix()
    t = torch.tensor([-1.2, 0.4, 2.8])
    x_tf = x @ R.T + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_tf, edge_index, edge_attr, atomic_masses, batch)

    if task == 0:
        assert_scalar_invariant(out_raw, out_tf)
    elif task == 1:
        assert_vector_rotates(out_raw, out_tf, R)
    elif task == 2:
        assert_tensor_rotates(out_raw, out_tf, R)
    else:
        s_raw, v_raw, t_raw = out_raw
        s_tf, v_tf, t_tf = out_tf
        assert_scalar_invariant(s_raw, s_tf)
        assert_vector_rotates(v_raw, v_tf, R)
        assert_tensor_rotates(t_raw, t_tf, R)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))