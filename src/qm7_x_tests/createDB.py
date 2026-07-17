# Code from QM7-X Zenodo https://zenodo.org/records/4288677
# QM7-X paper: https://www.nature.com/articles/s41597-021-00812-2

import schnetpack
import numpy as np
import h5py
from sys import stdout
from ase.atoms import Atoms

## function to exclude duplicates from QM7-X dataset
def removing_duplicates(IDs):

  DupMols = []
  for line in open('DupMols.dat', 'r'):
    DupMols.append(line.rstrip('\n'))

  for IDconf in IDs:
    if IDconf in DupMols:
      IDs.remove(IDconf)
      stmp = IDconf[:-3]
      for ii in range(1,101):
        IDs.remove(stmp+'d'+str(ii))

  return IDs

## define name of database
dbname = 'QM7X.db'

## initialize database
dataset = schnetpack.data.AtomsData(dbname, available_properties=['Eat','EMBD']) #use this line depending on SchNetPack version 
#dataset = schnetpack.data.AtomsData(dbname)

# atom energies
EPBE0_atom = {6:-1027.592489146, 17:-12516.444619523, 1:-13.641404161, \
              7:-1484.274819088, 8:-2039.734879322, 16:-10828.707468187}

## buffers for molecules and properties
atoms_buffer = []
property_buffer = []
stdout.write('\n')

## for all sets of molecules (representing individual files):
set_ids = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']
#set_ids = ['8000']
for setid in set_ids:
    ## load HDF5 file
    fMOL = h5py.File(setid+'.hdf5', 'r')
    
    ## get IDs of HDF5 files and loop through
    mol_ids = list(fMOL.keys())
    for molid in mol_ids:
        stdout.write('Current molecule: '+molid+'\n  Conformations:')
        
        ## get IDs of individual configurations/conformations of molecule
        conf_ids = list(fMOL[molid].keys())

        ## use this option if you want to exclude duplicates
#        conf_ids = removing_duplicates(conf_ids)

        for confid in conf_ids:
            
            ## get atomic positions and numbers and add to molecules buffer
            xyz = np.array(fMOL[molid][confid]['atXYZ'])
            Z = np.array(fMOL[molid][confid]['atNUM'])
            atoms_buffer.append( Atoms(Z, xyz) )
            
            ## get quantum mechanical properties and add them to properties buffer
            ## The user decides the properties to save in the DB file (see README.txt)
            force = list(fMOL[molid][confid]['totFOR'])
            Eatoms = sum([ EPBE0_atom[zi] for zi in Z ])
            Eat = float(list(fMOL[molid][confid]['eAT'])[0])
            EPBE0 = float(list(fMOL[molid][confid]['ePBE0'])[0])
            EMBD = float(list(fMOL[molid][confid]['eMBD'])[0])
            C6 = float(list(fMOL[molid][confid]['mC6'])[0])
            POL = float(list(fMOL[molid][confid]['mPOL'])[0])
            HLGAP = float(list(fMOL[molid][confid]['HLgap'])[0])
            DIP = float(list(fMOL[molid][confid]['DIP'])[0])
            property_buffer.append( {'forces':np.array(force),\
                                     'EPBE0':np.array([EPBE0]), \
                                     'Eat':np.array([Eat]), \
                                     'EMBD':np.array([EMBD]), \
                                     'C6':np.array([C6]), \
                                     'POL':np.array([POL]), \
                                     'HLGAP':np.array([HLGAP]), \
                                     'DIP':np.array([DIP]) })

        stdout.write('Molecule '+molid+' done.\n')
        
    
## create database file
stdout.write('\n\n=====================================================\n\n')
stdout.write('  Gathering data done. Writing database...\n')
dataset.add_systems(atoms_buffer, property_buffer)
stdout.write('\n  Database "'+dbname+'" written. \n\n')
stdout.write('=====================================================\n\n')
##--EOF--##
