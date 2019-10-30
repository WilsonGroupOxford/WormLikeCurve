#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:48:26 2019

@author: matthew-bailey
"""

from collections import defaultdict
import numpy as np

def parse_molecule_topology(filename: str):
    """
    Extracts atom, molecule and position information
    from a LAMMPS data file.
    :param filename: the name of the lammps file to open
    :return atoms: a dictionary of atoms, with atom ids as keys
    and the values are a dictionary of the type and position.
    :return molecs: a dictionary of molecules, with molecule ids
    as keys and values a list of atoms in that molecule.
    :return bonds: a list of pairs, representing atom ids
    at each end of the bonds.
    """
    bonds_mode = False
    atoms_mode = False
    molecules = defaultdict(list)
    atoms = defaultdict(dict)
    bonds = []
    with open(filename, "r") as fi:
        for line in fi.readlines():
            if not line:
                continue
            if "Atoms" in line:
                atoms_mode = True
                bonds_mode = False
                continue
            if "Bonds" in line:
                atoms_mode = False
                bonds_mode = True
                continue
            if "Angles" in line:
                # To be implemented
                atoms_mode = False
                bonds_mode = False
                continue

            if atoms_mode:
                try:
                    atom_id, molec_id, atom_type, x, y, z = line.split()
                except ValueError:
                    if line == "\n":
                        continue
                    print("Could not read line:", line,
                          "expected form: atom_id, molec_id, type, x, y, z")
                    continue
                atom_id = int(atom_id)
                molec_id = int(molec_id)
                atom_type = int(atom_type)
                x, y, z = float(x), float(y), float(z)
                atoms[atom_id] = {"type": atom_type,
                                  "pos":np.array([x, y, z])}
                molecules[molec_id].append(atom_id)
            if bonds_mode:
                try:
                    bond_id, bond_type, atom_a, atom_b = line.split()
                except ValueError:
                    if line == "\n":
                        continue
                    print("Could not read bond line:", line,
                          "Expected form: bond_id, bond_type, a, b")
                    continue
                bonds.append([int(atom_a), int(atom_b)])
    return atoms, molecules, bonds