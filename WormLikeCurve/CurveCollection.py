#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:26:42 2019

@author: matthew-bailey
"""
import numpy as np

class CurveCollection:
    """
    Helper class to operate on multiple curve objects,
    including plotting them and writing them to a lammps file.
    """

    def __init__(self, curves=None):
        if not curves:
            curves = []
        try:
            # Test if the curves are iterable.
            _ = iter(curves)
        except TypeError:
            curves = [curves]
        self.curves = curves
        
    def __iter__(self):
        yield from self.curves
        
    def __len__(self):
        return len(self.curves)
    
    def __getitem__(self, key):
        return self.curves[key]

    def __setitem__(self, key, value):
        self.curves[key] = value

    def __delitem__(self, key):
        del self.curves[key]
  
    @property
    def num_atoms(self):
        num_atoms = 0
        for curve in self.curves:
            num_atoms += curve.num_atoms
        return num_atoms
            
    @property
    def num_bonds(self):
        num_bonds = 0
        for curve in self.curves:
            num_bonds += curve.num_bonds
        return num_bonds
            
    @property
    def num_angles(self):
        num_angles = 0
        for curve in self.curves:
            num_angles += curve.num_angles
        return num_angles
    
    @property
    def positions(self):
        return np.concatenate([curve.positions for curve in self.curves],
                              axis=0)
            
    def append(self, curve):
        """
        Add a new curve into our collection.
        :param curve: the curve to be added
        """
        self.curves.append(curve)
 
    def plot_onto(self,
                  ax,
                  kwarg_list = [],
                  fit_edges: bool = True,):
        """
        Plots these polymers as a collection of lines, detailed by kwargs
        into the provided axis.
        :param ax: a matplotlib axis object
        :param fit_edges: should the axis size be rearranged to fit
        :param **kwargs: keyword arguments to be passed to matplotlib
        """
        length_difference = len(self) - len(kwarg_list)
        if length_difference > 0:
            kwarg_list.extend([{} for _ in range(length_difference)])
        elif length_difference < 0:
            print("More kwargs provided than there are curves." + 
                  "Ignoring excess")
        
        for i, curve in enumerate(self.curves):
            # Don't pass on edge fitting because we'll
            # do it here.
            curve.plot_onto(ax,
                            fit_edges=False,
                            **kwarg_list[i])

        if fit_edges:
            min_x, min_y = np.min(self.positions, axis=0)
            max_x, max_y = np.max(self.positions, axis=0)
            min_corner = (min(min_x, min_y) * 1.1) - 0.1
            max_corner = (max(max_x, max_y) * 1.1) + 0.1
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)

    def to_lammps(self,
                  filename: str):
        with open(filename, "w") as fi:
            # Header section
            fi.write("Polymer file\n\n")
            fi.write(f"\t {self.num_atoms}\t atoms\n")
            fi.write(f"\t {self.num_bonds}\t bonds\n")
            fi.write(f"\t {self.num_angles}\t angles\n\n")

            atom_types = set([])
            for curve in self.curves:
                atom_types = atom_types.union(set(np.unique(curve.atom_types)))
            num_atom_types = len(atom_types)
            fi.write(f"\t {num_atom_types} \t atom types\n")
            fi.write("\t 1 \t bond types\n")
            fi.write("\t 1 \t angle types\n\n")

            min_x, min_y = np.min(self.positions, axis=0)
            max_x, max_y = np.max(self.positions, axis=0)
            min_corner = (min(min_x, min_y) * 1.1) - 0.1
            max_corner = (max(max_x, max_y) * 1.1) + 0.1

            fi.write(f"\t {min_corner:.3f} {max_corner:.3f} \t xlo xhi\n")
            fi.write(f"\t {min_corner:.3f} {max_corner:.3f} \t ylo yhi\n")
            fi.write(f"\t -1.0 1.0 \t zlo zhi\n\n")

            # Masses
            fi.write("Masses\n\n")
            for atom_type in atom_types:
                fi.write(f"\t {atom_type} \t 14.02\n")
            fi.write("\n")

            # Atom positions
            fi.write("Atoms\n\n")
            atom_id = 0
            for molecule_id, curve in enumerate(self.curves, 1):
                for i, atom in enumerate(curve.positions):
                    atom_id += 1
                    # format:
                    # atom_id molecule_id atom_type x y z
                    atom_type = curve.atom_types[i]
                    fi.write(f"\t {atom_id} \t {molecule_id} \t {atom_type}" +
                             f"\t {atom[0]:.3f} \t {atom[1]:.3f} \t 0.000\n")
            fi.write("\n")

            # Bonds
            fi.write("Bonds\n\n")
            bond_id = 0
            total_atoms = 1
            for curve in self.curves:
                print(curve.bonds)
                for atom_a, atom_b in curve.bonds:
                    bond_id += 1
                    # format:
                    # bond_id atom_1 atom_2
                    atom_a += total_atoms
                    atom_b += total_atoms
                    fi.write(f"\t {bond_id} \t 1 \t {atom_a} \t {atom_b}\n")
                # Skip over one atom because end of curves[0]
                # is not connected to start of curves[1]
                total_atoms += curve.num_atoms
 
            fi.write("\n")

            # Angles
            fi.write("Angles\n\n")
            atom_id = 0
            angle_id = 0
            for curve in self.curves:
                for _ in range(curve.num_angles):
                    atom_id += 1
                    angle_id += 1
                    fi.write(f"\t {angle_id} \t 1 \t {atom_id}" +
                             f"\t {atom_id + 1} \t {atom_id + 2}\n")
                # Skip head 2 to the central atom of the next angle
                atom_id += 2
            fi.write("\n")
