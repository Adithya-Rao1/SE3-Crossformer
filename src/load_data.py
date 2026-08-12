import os

import torch
from torch.utils.data import DataLoader as TorchLoader
import numpy as np

from torch_geometric.datasets import QM9, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# from atomic_datasets import GEOMDrugs

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

QM9_TARGET_CONVERSION = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    HAR2EV, 1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV,
])

class CustomQM9Dataset(InMemoryDataset):
    def __init__(
        self,
        root,
        sdf_file,
        csv_file,
        device,
        transform=None,
        pre_transform=None,
    ):
        self.device = device

        self.sdf_file = sdf_file
        self.csv_file = csv_file

        super().__init__(
            root,
            transform,
            pre_transform
        )

        self.data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False
        )

    @property
    def raw_file_names(self):
        return [
            os.path.basename(self.sdf_file),
            os.path.basename(self.csv_file)
        ]

    @property
    def processed_file_names(self):
        return ["processed_data.pt"]

    def process(self):

        df = pd.read_csv(self.csv_file)
        target_matrix = df.iloc[:, 4:].to_numpy(dtype=float)

        suppl = Chem.SDMolSupplier(
            self.sdf_file,
            removeHs=False,
            sanitize=True
        )

        data_list = []

        valid_mols = [
            (row_idx, mol)
            for row_idx, mol in enumerate(suppl)
            if mol is not None
        ]

        if len(valid_mols) != len(df):
            print(
                f"Warning: {len(df)} csv rows "
                f"but {len(valid_mols)} valid molecules."
            )

        for idx, (row_idx, mol) in enumerate(
            tqdm(valid_mols, desc="Processing QM9")
        ):
            if row_idx >= len(target_matrix):
                continue

            atom_features = []

            for atom in mol.GetAtoms():

                atom_features.append(
                    atom.GetAtomicNum(),
                )

            x = torch.tensor(
                atom_features,
                dtype=torch.float,
                device=self.device
            )

            rows = []
            cols = []
            edge_features = []

            for bond in mol.GetBonds():

                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                bond_type = bond.GetBondType()
                bond_onehot = [
                    1.0 if bond_type == t else 0.0 for t in BOND_TYPES
                ]

                rows.extend([i, j])
                cols.extend([j, i])

                edge_features.extend([
                    bond_onehot,
                    bond_onehot
                ])

            if len(rows) == 0:
                continue

            edge_index = torch.tensor(
                [rows, cols],
                dtype=torch.long,
                device=self.device
            )

            edge_attr = torch.tensor(
                edge_features,
                dtype=torch.float,
                device=self.device
            )

            conf = mol.GetConformer()

            pos = torch.tensor(
                conf.GetPositions(),
                dtype=torch.float,
                device=self.device
            )

            if pos.shape[0] != x.shape[0]:
                continue

            y = torch.tensor(
                target_matrix[row_idx],
                dtype=torch.float,
                device=self.device
            ).reshape(1, -1) * QM9_TARGET_CONVERSION.to(self.device)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=y,
                idx=torch.tensor([idx]),
            )

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in data_list
            ]

        data, slices = self.collate(data_list)

        torch.save(
            (data, slices),
            self.processed_paths[0]
        )

def to_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    return x

def sanitize_data(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = torch.from_numpy(value)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], np.ndarray):
                data[key] = [torch.from_numpy(v) for v in value]
            else:
                data[key] = value
        else:
            data[key] = value
    return data


def geom_collate_fn(batch):
    return [sanitize_data(data) for data in batch]

config = {
    "batch_size":20,
    "shuffle": True,
}

def load_data(dataset_type, config):
    assert dataset_type in ['qm9', 'zinc', 'geom'], "Available datasets are qm9, zinc, or geom."

    if dataset_type == "qm9" or dataset_type == "zinc":
        if dataset_type == "qm9":
            dataset = CustomQM9Dataset(
                    root="./data/qm1", 
                    sdf_file="./data/qm9/raw/gdb9.sdf", 
                    csv_file="./data/qm9/raw/gdb9.sdf.csv",
            )
# )
        elif dataset_type == "zinc":
            dataset = ZINC(root='./data/zinc')
        loader = DataLoader(dataset=dataset,
                            **config)
    """elif dataset_type == "geom":
        dataset = GEOMDrugs(root_dir="data/geom")
        loader = TorchLoader(dataset, **config, collate_fn=geom_collate_fn)"""
    
    return loader