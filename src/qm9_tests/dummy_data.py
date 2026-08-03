import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ATOM_TYPES = [1, 6, 7, 8, 9]
BOND_ORDERS = [1.0, 1.5, 2.0, 3.0]
NUM_TARGETS = 16  # mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom


def _make_dummy_molecule(num_atoms, generator):
    atom_idx = torch.randint(0, len(ATOM_TYPES), (num_atoms,), generator=generator)
    z = torch.tensor([ATOM_TYPES[i] for i in atom_idx.tolist()], dtype=torch.float)
    pos = torch.randn(num_atoms, 3, generator=generator)

    perm = torch.randperm(num_atoms, generator=generator)
    rows, cols, edge_features = [], [], []
    for k in range(1, num_atoms):
        child = perm[k].item()
        parent = perm[torch.randint(0, k, (1,), generator=generator).item()].item()
        bond_order = BOND_ORDERS[torch.randint(0, len(BOND_ORDERS), (1,), generator=generator).item()]
        rows.extend([child, parent])
        cols.extend([parent, child])
        edge_features.extend([[bond_order], [bond_order]])

    num_extra = max(0, num_atoms // 3)
    for _ in range(num_extra):
        i, j = torch.randint(0, num_atoms, (2,), generator=generator).tolist()
        if i == j:
            continue
        bond_order = BOND_ORDERS[torch.randint(0, len(BOND_ORDERS), (1,), generator=generator).item()]
        rows.extend([i, j])
        cols.extend([j, i])
        edge_features.extend([[bond_order], [bond_order]])

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    y = torch.randn(1, NUM_TARGETS, generator=generator)

    return Data(x=z, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)


def _make_dummy_dataset(num_molecules, generator, min_atoms=6, max_atoms=15):
    data_list = []
    for _ in range(num_molecules):
        num_atoms = torch.randint(min_atoms, max_atoms + 1, (1,), generator=generator).item()
        data_list.append(_make_dummy_molecule(num_atoms, generator))
    return data_list


def load_qm9_dummy(batch_size=16, r="./data", device=torch.device("cpu"), num_batches=6, seed=0):
    generator = torch.Generator().manual_seed(seed)

    train_batches = max(1, num_batches)
    val_batches = max(1, num_batches // 3)
    test_batches = max(1, num_batches // 3)

    train_list = _make_dummy_dataset(batch_size * train_batches, generator)
    val_list = _make_dummy_dataset(batch_size * val_batches, generator)
    test_list = _make_dummy_dataset(batch_size * test_batches, generator)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader