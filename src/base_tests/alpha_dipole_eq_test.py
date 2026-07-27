import pytest
import torch

from src.se3_crossformer.se3_utils import RadialNetworkGSFB
from src.qme14s_tests.train_qme14s import SE3PolarizabilityInterTransformer, SE3PolarizabilityIntraTransformer, SE3DeDipoleInterTransformer, SE3DeDipoleIntraTransformer

torch.manual_seed(0)

ATOL = 1e-8
RTOL = 1e-8

POLARIZABILITY_MODEL_CLASSES = [SE3PolarizabilityInterTransformer, SE3PolarizabilityIntraTransformer]
DEDIPOLE_MODEL_CLASSES = [SE3DeDipoleInterTransformer, SE3DeDipoleIntraTransformer]


def random_rotation_matrix(dtype=torch.double, generator=None) -> torch.Tensor:
    A = torch.randn(3, 3, dtype=torch.double, generator=generator)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R).sign()
    Q = Q * d
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def make_toy_graph(n_atoms=12, n_atom_types=5, num_parts=4, seed=0):
    g = torch.Generator().manual_seed(seed)

    x = torch.randn(n_atoms, 3, generator=g).double()

    atom_type_idx = torch.randint(0, n_atom_types, (n_atoms,), generator=g)
    node_features = torch.nn.functional.one_hot(
        atom_type_idx, num_classes=n_atom_types
    ).double()

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


def make_two_graph_batch(n_atom_types=5, num_parts=4, n_atoms_a=10, n_atoms_b=14,
                          seed_a=1, seed_b=2):
    nf_a, x_a, ei_a, ea_a, am_a, _ = make_toy_graph(
        n_atoms=n_atoms_a, n_atom_types=n_atom_types, num_parts=num_parts, seed=seed_a
    )
    nf_b, x_b, ei_b, ea_b, am_b, _ = make_toy_graph(
        n_atoms=n_atoms_b, n_atom_types=n_atom_types, num_parts=num_parts, seed=seed_b
    )

    node_features = torch.cat([nf_a, nf_b], dim=0)
    x = torch.cat([x_a, x_b], dim=0)
    atomic_masses = torch.cat([am_a, am_b], dim=0)
    edge_index = torch.cat([ei_a, ei_b + n_atoms_a], dim=1)
    edge_attr = torch.cat([ea_a, ea_b], dim=0)
    batch = torch.cat([
        torch.zeros(n_atoms_a, dtype=torch.long),
        torch.ones(n_atoms_b, dtype=torch.long),
    ])
    return node_features, x, edge_index, edge_attr, atomic_masses, batch


def build_model(model_cls, in_features, num_parts=4, max_degree=2):
    model = model_cls(
        radial_net=RadialNetworkGSFB,
        in_features=in_features,
        max_degree=max_degree,
        num_layers=2,
        feature_dim=16,
        hidden_dim=32,
        num_parts=num_parts,
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


def assert_tensor_rotates(out_raw, out_tf, R):
    expected = torch.einsum("ij,bjk,lk->bil", R, out_raw, R)
    torch.testing.assert_close(out_tf, expected, atol=ATOL, rtol=RTOL)


def assert_tensor_invariant(out_raw, out_tf):
    torch.testing.assert_close(out_raw, out_tf, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("model_cls", POLARIZABILITY_MODEL_CLASSES)
def test_polarizability_rotation_equivariance(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=42
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    x_rot = x @ R.T

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_rot, edge_index, edge_attr, atomic_masses, batch)

    assert out_raw.shape == (1, 3, 3)
    assert_tensor_rotates(out_raw, out_tf, R)


@pytest.mark.parametrize("model_cls", POLARIZABILITY_MODEL_CLASSES)
def test_polarizability_translation_invariance(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=7
    )
    model = build_model(model_cls, in_features=n_atom_types)

    t = torch.tensor([3.7, -2.1, 5.5], dtype=torch.double)
    x_shifted = x + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_shifted, edge_index, edge_attr, atomic_masses, batch)

    assert_tensor_invariant(out_raw, out_tf)


@pytest.mark.parametrize("model_cls", POLARIZABILITY_MODEL_CLASSES)
def test_polarizability_combined_rotation_and_translation(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=14, n_atom_types=n_atom_types, num_parts=4, seed=99
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    t = torch.tensor([-1.2, 0.4, 2.8], dtype=torch.double)
    x_tf = x @ R.T + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_tf, edge_index, edge_attr, atomic_masses, batch)

    assert_tensor_rotates(out_raw, out_tf, R)


@pytest.mark.parametrize("model_cls", POLARIZABILITY_MODEL_CLASSES)
def test_polarizability_output_is_symmetric(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=13
    )
    model = build_model(model_cls, in_features=n_atom_types)
    out = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    torch.testing.assert_close(out, out.transpose(-1, -2), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("model_cls", POLARIZABILITY_MODEL_CLASSES)
def test_polarizability_batched_combined_transform(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_two_graph_batch(
        n_atom_types=n_atom_types, num_parts=4
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.double)
    x_tf = x @ R.T + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_tf, edge_index, edge_attr, atomic_masses, batch)

    assert out_raw.shape == (2, 3, 3)
    assert_tensor_rotates(out_raw, out_tf, R)

@pytest.mark.parametrize("model_cls", DEDIPOLE_MODEL_CLASSES)
def test_dedipole_rotation_equivariance(model_cls):
    n_atom_types = 5
    n_atoms = 12
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=n_atoms, n_atom_types=n_atom_types, num_parts=4, seed=42
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    x_rot = x @ R.T

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_rot, edge_index, edge_attr, atomic_masses, batch)

    assert out_raw.shape == (n_atoms, 3, 3)
    assert_tensor_rotates(out_raw, out_tf, R)


@pytest.mark.parametrize("model_cls", DEDIPOLE_MODEL_CLASSES)
def test_dedipole_translation_invariance(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=12, n_atom_types=n_atom_types, num_parts=4, seed=7
    )
    model = build_model(model_cls, in_features=n_atom_types)

    t = torch.tensor([3.7, -2.1, 5.5], dtype=torch.double)
    x_shifted = x + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_shifted, edge_index, edge_attr, atomic_masses, batch)

    assert_tensor_invariant(out_raw, out_tf)


@pytest.mark.parametrize("model_cls", DEDIPOLE_MODEL_CLASSES)
def test_dedipole_combined_rotation_and_translation(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_toy_graph(
        n_atoms=14, n_atom_types=n_atom_types, num_parts=4, seed=99
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    t = torch.tensor([-1.2, 0.4, 2.8], dtype=torch.double)
    x_tf = x @ R.T + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_tf, edge_index, edge_attr, atomic_masses, batch)

    assert_tensor_rotates(out_raw, out_tf, R)


@pytest.mark.parametrize("model_cls", DEDIPOLE_MODEL_CLASSES)
def test_dedipole_batched_combined_transform(model_cls):
    n_atom_types = 5
    node_features, x, edge_index, edge_attr, atomic_masses, batch = make_two_graph_batch(
        n_atom_types=n_atom_types, num_parts=4
    )
    model = build_model(model_cls, in_features=n_atom_types)

    R = random_rotation_matrix()
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.double)
    x_tf = x @ R.T + t

    out_raw = run(model, node_features, x, edge_index, edge_attr, atomic_masses, batch)
    out_tf = run(model, node_features, x_tf, edge_index, edge_attr, atomic_masses, batch)

    assert out_raw.shape == (node_features.shape[0], 3, 3)
    assert_tensor_rotates(out_raw, out_tf, R)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))