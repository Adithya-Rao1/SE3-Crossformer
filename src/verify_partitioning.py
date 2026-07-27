import argparse
import math
import numpy as np
import torch
from typing import List, Tuple

from src.se3_crossformer.spectral_partition import spectral_partition

def get_functional_group_boundaries(mol) -> List[frozenset]:
    """
    TODO: Choose an extensive SMARTS library / fragment list
    """
    try:
        from rdkit.Chem import MolFromSmarts
    except ImportError:
        raise ImportError("RDKit is required: pip install rdkit")

    SMARTS = {
        "alcohol":    "[OX2H]",
        "amine":      "[NX3;H2,H1;!$(NC=O)]",
        "carboxyl":   "[CX3](=O)[OX2H1]",
        "aldehyde":   "[CX3H1](=O)[#6]",
        "ketone":     "[#6][CX3](=O)[#6]",
        "amide":      "[NX3][CX3](=[OX1])[#6]",
        "ether":      "[OD2]([#6])[#6]",
        "halide":     "[F,Cl,Br,I]",
        "nitro":      "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "nitrile":    "[NX1]#[CX2]",
    }

    boundaries = []
    for name, sma in SMARTS.items():
        pat = MolFromSmarts(sma)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        for match in matches:
            boundaries.append(frozenset(match))

    return boundaries


def mol_to_graph(mol) -> Tuple[torch.Tensor, int]:
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    if not edges:
        return torch.zeros(2, 0, dtype=torch.long), mol.GetNumAtoms()
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, mol.GetNumAtoms()

def knn_partition(
    pos: np.ndarray,    
    num_parts: int,
    k: int,
) -> np.ndarray:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_parts, n_init=10, random_state=0)
    return kmeans.fit_predict(pos)

def partition_boundary_distance(
    partition: np.ndarray,       
    reference_groups: List[frozenset],
    N: int,
) -> float:
    """
    TODO: May need more rigorous/meaningful "boundary distance" metric
    """
    if not reference_groups:
        return float("nan")

    distances = []
    for group in reference_groups:
        best_jaccard = 0.0
        unique_clusters = set(partition.tolist())
        for c in unique_clusters:
            cluster_set = frozenset(np.where(partition == c)[0].tolist())
            inter = len(group & cluster_set)
            union = len(group | cluster_set)
            jaccard = inter / union if union > 0 else 0.0
            best_jaccard = max(best_jaccard, jaccard)
        distances.append(1.0 - best_jaccard)  

    return float(np.mean(distances))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_molecules", type=int, default=100)
    parser.add_argument("--num_parts",   type=int, default=6)
    parser.add_argument("--k_nn",        type=int, default=6,  help="Clusters for kNN baseline")
    parser.add_argument("--data_root",   type=str, default="./data")
    args = parser.parse_args()

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit required: pip install rdkit")

    try:
        from torch_geometric.datasets import ZINC
    except ImportError:
        raise ImportError("torch_geometric required for ZINC-250K dataset.")

    print(f"Loading ZINC-250K (sampling {args.n_molecules} molecules)...")
    dataset = ZINC(root=args.data_root, subset=False, split="train")

    spectral_dists = []
    knn_dists      = []
    skipped        = 0

    for i, data in enumerate(dataset):
        if i >= args.n_molecules:
            break

        # TODO: ZINC torch_geometric data does not store SMILES directly in all versions. Add feature to get SMILES
        N = data.num_nodes
        if N < args.num_parts:
            skipped += 1
            continue

        edge_index = data.edge_index  

        try:
            sp_labels = spectral_partition(edge_index, N, args.num_parts).numpy()
        except Exception as e:
            print(f"  Molecule {i}: spectral partition failed ({e}), skipping.")
            skipped += 1
            continue

        # TODO: Use actual 3D conformer positions when available.
        proxy_pos = np.random.randn(N, 3)   # TODO: Replace with real conformer coords
        knn_labels = knn_partition(proxy_pos, args.num_parts, args.k_nn)

        # TODO: Load SMILES and call get_functional_group_boundaries(mol).
        reference_groups: List[frozenset] = []

        if not reference_groups:
            skipped += 1
            continue

        sp_d  = partition_boundary_distance(sp_labels,  reference_groups, N)
        knn_d = partition_boundary_distance(knn_labels, reference_groups, N)

        spectral_dists.append(sp_d)
        knn_dists.append(knn_d)

    print(f"\nProcessed: {len(spectral_dists)} molecules  (skipped: {skipped})")

    if len(spectral_dists) < 2:
        print("Not enough data for CI. Load SMILES to enable full experiment.")
        return

    diffs = np.array(knn_dists) - np.array(spectral_dists)
    n     = len(diffs)
    mean  = np.mean(diffs)
    std   = np.std(diffs, ddof=1)
    from scipy import stats
    t_crit   = stats.t.ppf(0.975, df=n - 1)
    half_w   = t_crit * std / math.sqrt(n)
    ci_lo    = mean - half_w
    ci_hi    = mean + half_w

    print(f"\nMean distance (kNN):      {np.mean(knn_dists):.4f}")
    print(f"Mean distance (spectral): {np.mean(spectral_dists):.4f}")
    print(f"Mean difference (kNN - spectral): {mean:.4f}")
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    if ci_lo > 0:
        print("=> 0 NOT in CI: spectral partitioning is significantly better.")
    elif ci_hi < 0:
        print("=> 0 NOT in CI: kNN is significantly better (unexpected).")
    else:
        print("=> 0 in CI: no significant difference detected.")


if __name__ == "__main__":
    main()