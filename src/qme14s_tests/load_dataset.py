import h5py
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Code from QMe14S authors

#md
def read_hdf5_files(num_files):
    data_list = []
    for file_idx in tqdm(range(num_files), desc="Reading HDF5 files", unit="file"):
        filename = f'data_chunk_{file_idx}.h5'
        with h5py.File(filename, 'r') as h5file:
            for group_name in h5file:
                group = h5file[group_name]
                
                # Read edge_index
                edge_index = torch.tensor(group['edge_index'][:])
                
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read z
                z = torch.tensor(group['z'][:])
                
                # Read dipole
                dipole = torch.tensor(group['dipole'][:])
                
                # Read force
                force = torch.tensor(group['force'][:])
                
                # Read atomization_energy
                atomization_energy = group.attrs['atomization_energy']
                
                # Read smile
                smile = group.attrs['smile']
                
                # Construct Data object
                data = Data(
                    edge_index=edge_index,
                    pos=pos,
                    z=z,
                    dipole=dipole,
                    force=force,
                    atomization_energy=atomization_energy,
                    smile=smile,
                )
                
                # Add to data_list
                data_list.append(data)
    return data_list

"""# Read all files
num_files = 6  # Set based on the actual number of files
data_list = read_hdf5_files(num_files)

print(f"Loaded {len(data_list)} Data objects.")"""

# qme14_single_point.h5
def read_hdf5_file(filename):
    data_list = []
    with h5py.File(filename, 'r') as h5file:
        for group_name in tqdm(h5file.keys(), desc="Reading Data", unit="item"):
            group = h5file[group_name]
            try:
                # Read edge_index
                edge_index = torch.tensor(group['edge_index'][:])
                
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read z
                z = torch.tensor(group['z'][:])
                
                # Read atomization_energy
                atomization_energy = group.attrs.get('atomization_energy', None)
                
                # Read dipole
                dipole = torch.tensor(group['dipole'][:])
                
                # Read polar
                polar = torch.tensor(group['polar'][:])
                
                # Read dedipole
                dedipole = torch.tensor(group['dedipole'][:])
                
                # Read Hi
                Hi = torch.tensor(group['Hi'][:])
                
                # Read Hij
                Hij = torch.tensor(group['Hij'][:])
                
                # Read smile
                smile = group.attrs.get('smile', None)
                
                # Read r2
                r2 = group.attrs.get('r2', None)
                
                # Read quadrupole
                quadrupole = torch.tensor(group['quadrupole'][:])
                
                # Read octapole
                octapole = torch.tensor(group['octapole'][:])
                
                # Read force
                force = torch.tensor(group['force'][:])
                
                # Create Data object
                data = Data(
                    edge_index=edge_index,
                    pos=pos,
                    z=z,
                    atomization_energy=atomization_energy,
                    dipole=dipole,
                    polar=polar,
                    dedipole=dedipole,
                    Hi=Hi,
                    Hij=Hij,
                    smile=smile,
                    r2=r2,
                    quadrupole=quadrupole,
                    octapole=octapole,
                    force=force,
                )
                
                # Add to list
                data_list.append(data)
            
            except KeyError as e:
                print(f"Missing key {e} in group {group_name}. Skipping this group.")
                continue
            
            except Exception as e:
                print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                continue
                
    return data_list

"""filename = 'QMe14S_single_point.h5'  # HDF5 filename
data_list = read_hdf5_file(filename)

print(f"Successfully loaded {len(data_list)} Data objects.")"""


# opt_186102.h5
def read_hdf5_file(filename):
    data_list = []
    with h5py.File(filename, 'r') as h5file:
        for group_name in tqdm(h5file.keys(), desc="Reading Data", unit="item"):
            group = h5file[group_name]
            try:
                # Read edge_index
                edge_index = torch.tensor(group['edge_index'][:])
                
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read smile
                smile = group.attrs.get('smile', None)
                
                # Read z
                z = torch.tensor(group['z'][:])
                
                # Read quadrupole
                quadrupole = torch.tensor(group['quadrupole'][:])
                
                # Read octapole
                octapole = torch.tensor(group['octapole'][:])
                
                # Read npacharge
                npacharge = torch.tensor(group['npacharge'][:])
                
                # Read dipole
                dipole = torch.tensor(group['dipole'][:])
                
                # Read polar
                polar = torch.tensor(group['polar'][:])
                
                # Read hyperpolar
                hyperpolar = torch.tensor(group['hyperpolar'][:])
                
                # Read Hij
                Hij = torch.tensor(group['Hij'][:])
                
                # Read Hii
                Hii = torch.tensor(group['Hii'][:])
                
                # Read dedipole
                dedipole = torch.tensor(group['dedipole'][:])
                
                # Read depolar
                depolar = torch.tensor(group['depolar'][:])
                
                # Create Data object
                data = Data(
                    edge_index=edge_index,
                    pos=pos,
                    smile=smile,
                    z=z,
                    quadrupole=quadrupole,
                    octapole=octapole,
                    npacharge=npacharge,
                    dipole=dipole,
                    polar=polar,
                    hyperpolar=hyperpolar,
                    Hij=Hij,
                    Hii=Hii,
                    dedipole=dedipole,
                    depolar=depolar,
                )
                
                # Add to list
                data_list.append(data)
            
            except KeyError as e:
                print(f"Missing key {e} in group {group_name}. Skipping this group.")
                continue
            
            except Exception as e:
                print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                continue
                
    return data_list

# Use the read function
filename = '/home/ubuntu/se3-crossformer-data/qme14s_data/OPT_186102.h5'  # HDF5 filename
data_list = read_hdf5_file(filename)

print(f"Successfully loaded {len(data_list)} Data objects.")


# Hessian_opt.h5
def read_hdf5_file(filename):
    data_list = []
    with h5py.File(filename, 'r') as h5file:
        for group_name in tqdm(h5file.keys(), desc="Reading Data", unit="item"):
            group = h5file[group_name]
            try:
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read smiles
                smiles = group.attrs.get('smiles', None)
                
                # Read hessian
                hessian = torch.tensor(group['hessian'][:])
                
                # Create Data object
                data = Data(
                    pos=pos,
                    smiles=smiles,
                    hessian=hessian,
                )
                
                # Add to list
                data_list.append(data)
            
            except KeyError as e:
                print(f"Missing key {e} in group {group_name}. Skipping this group.")
                continue
            
            except Exception as e:
                print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                continue
                
    return data_list

"""# Use the read function
filename = 'Hessian_opt.h5'  # HDF5 filename
data_list = read_hdf5_file(filename)

print(f"Successfully loaded {len(data_list)} Data objects.")"""


# Hessian_single_point.h5
def read_hessian_hdf5_file(filename):
    data_list = []
    with h5py.File(filename, 'r') as h5file:
        for group_name in tqdm(h5file.keys(), desc="Reading Data", unit="item"):
            group = h5file[group_name]
            try:
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read z
                z = torch.tensor(group['z'][:])
                
                # Read hessian
                hessian = torch.tensor(group['hessian'][:])
                
                # Read smile
                smile = group.attrs.get('smile', None)
                
                # Create Data object
                data = Data(
                    pos=pos,
                    z=z,
                    hessian=hessian,
                    smile=smile
                )
                
                # Add to list
                data_list.append(data)
            
            except KeyError as e:
                print(f"Missing key {e} in group {group_name}. Skipping this group.")
                continue
            
            except Exception as e:
                print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                continue
                
    return data_list

"""# Use the read function
filename = 'Hessian_single_point.h5'  # HDF5 filename
data_list = read_hessian_hdf5_file(filename)

print(f"Successfully loaded {len(data_list)} Data objects.")"""


# nmr.h5
def read_nmr_hdf5_file(filename):
    data_list = []
    with h5py.File(filename, 'r') as h5file:
        for group_name in tqdm(h5file.keys(), desc="Reading Data", unit="item"):
            group = h5file[group_name]
            try:
                # Read pos
                pos = torch.tensor(group['pos'][:])
                
                # Read z
                z = torch.tensor(group['z'][:])
                
                # Read nst
                nst = torch.tensor(group['nst'][:])
                
                # Read nst_iso
                nst_iso = torch.tensor(group['nst_iso'][:])
                
                # Read smile
                smile = group.attrs.get('smile', None)
                
                # Create Data object
                data = Data(
                    pos=pos,
                    z=z,
                    nst=nst,
                    nst_iso=nst_iso,
                    smile=smile
                )
                
                # Add to list
                data_list.append(data)
            
            except KeyError as e:
                print(f"Missing key {e} in group {group_name}. Skipping this group.")
                continue
            
            except Exception as e:
                print(f"Unexpected error in group {group_name}: {e}. Skipping this group.")
                continue
                
    return data_list

""" Use the read function
filename = 'nmr.h5'  # HDF5 filename
data_list = read_nmr_hdf5_file(filename)

print(f"Successfully loaded {len(data_list)} Data objects.")"""