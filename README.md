# SE(3)-Crossformer

**SE(3)-Crossformer** is a graph transformer for molecular representation learning that extends the SE(3)-Transformer by incorporating hierarchical geometric attention between both atoms and subgraphs. The model is tested on molecular property and vibrational spectroscopy tasks.

Unlike conventional SE(3)-equivariant graph neural networks that primarily model interactions between nearby atoms, SE(3)-Crossformer introduces a second level of message passing between graph partitions, allowing the network to capture long-range geometric and chemical interactions while preserving exact SE(3)-equivariance.

---

## Motivation

Many molecular properties arise from interactions that extend well beyond the local atomic neighborhood such as long-range Coulombic attractions and Van der Waals forces. Existing architectures such as the SE(3)-Transformer rely on nearest-neighbor graphs and local attention, which effectively model short-range interactions but often struggle to represent global effects induced by distant functional groups or larger molecular substructures.

SE(3)-Crossformer addresses this limitation by constructing a hierarchical representation of the molecular graph:

- **Atom-level attention** captures local geometric interactions.
- **Partition-level attention** models communication between chemically meaningful regions of the molecule.
- **Cross-attention** broadcasts global neighborhood information back to individual atoms.

This hierarchy enables information to propagate over significantly longer spatial scales without requiring dense all-to-all attention.

---

## Mathematical Foundation

SE(3)-Crossformer is built upon the representation theory of the Special Euclidean group SE(3). Rather than treating node features as ordinary vectors, each feature is represented as a collection of irreducible representations (irreps) of SO(3):

```math
\mathbf{f}
=
\bigoplus_{\ell=0}^{L}
\mathbf{f}^{(\ell)},
```

where each degree $\ell$ transforms according to the corresponding Wigner $D$-matrix under rotation.

Equivariant linear operators are parameterized using spherical harmonics and Clebsch–Gordan tensor products,

```math
\mathbf{W}^{\ell k}(\mathbf{x})
=
\sum_{J=|\ell-k|}^{\ell+k}
\phi_J(\|\mathbf{x}\|)
\,
\mathbf{Y}_J(\hat{\mathbf{x}})
\,
\mathbf{Q}^{\ell k},
```

allowing learned filters to satisfy

```math
\mathbf{W}(R\mathbf{x})
=
D(R)\,
\mathbf{W}(\mathbf{x})\,
D(R)^{-1}
```

for every rotation $R \in SO(3)$.

Consequently, every attention layer, feature update, and message passing operation transforms consistently under arbitrary three-dimensional rotations and translations.

---

## Hierarchical Attention

SE(3)-Crossformer extends the original SE(3)-Transformer with a second attention hierarchy.

Instead of viewing the molecule solely as a collection of atoms, the molecular graph is partitioned into multiple subgraphs using spectral graph partitioning. Each partition produces an equivariant latent representation describing the local chemical environment.

The model performs three complementary attention operations:

- **Atom → Atom:** Local SE(3)-equivariant self-attention within each partition.
- **Partition → Partition:** Communication between neighborhood representations through an equivariant message-passing network.
- **Partition → Atom:** Cross-attention that injects global structural information back into individual atomic representations.

This hierarchical formulation allows distant regions of the molecule to influence one another without requiring prohibitively expensive global atom-to-atom attention.

---

## Representation Learning

Different irreducible representations naturally encode different physical quantities.

- **Type-0 ($\ell=0$)** features represent scalar quantities such as molecular energies.
- **Type-1 ($\ell=1$)** features represent vector-valued quantities such as dipole moments.
- **Type-2 ($\ell=2$)** features represent rank-two tensor quantities such as polarizability tensors.

This makes the architecture well suited for predicting quantum mechanical observables while respecting their underlying transformation laws.

---

## Numerical Equivariance

A significant portion of this project focused on verifying exact numerical equivariance.

During development, multiple sources of equivariance error were identified and corrected, including:

- Improper bias terms in equivariant linear operators.
- Incorrect application of linear maps across irrep dimensions.
- Inconsistencies between real and complex spherical harmonic conventions.
- Incorrect Clebsch–Gordan basis transformations.

The final implementation passes exhaustive unit tests for

- Rotation equivariance
- Translation invariance
- Combined SE(3) transformations

with numerical errors on the order of

```math
10^{-10},
```

approaching floating-point precision.

---

## Current Applications

The current implementation supports molecular representation learning on standard benchmark datasets such as **QM9**, with ongoing work extending the architecture toward molecular dynamics and vibrational spectroscopy prediction.

Future work includes incorporating temporal information from molecular dynamics trajectories to directly model infrared and Raman spectra while preserving SE(3)-equivariance across space and time.