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

    def rescale(self, scale_factor: float):
        """
        Rescale the entire collection by a given factor.

        Parameters
        ----------
        scale_factor : float
            The amount to scale the box up by.

        Returns
        -------
        None.

        """
        for curve in self:
            curve.rescale(scale_factor)

    def translate(self, translation: np.array):
        """
        Translate the entire collection by a given vector.

        Parameters
        ----------
        scale_factor : float
            The amount to scale the box up by.

        Returns
        -------
        None.

        """
        assert translation.shape[0] == 2
        for curve in self:
            curve.translate(translation)

    def apply_transformation_matrix(self, transformation_matrix):
        """
        Apply an arbitrary transformation to each polymer

        This transformation is a 2x2 matrix with real
        components that represents a 2D scale, skew etc.
        Rotation and isotropic scaling matrices are specialised in
        rotate() and rescale() functions with some safety checks.
        This does not recentre on the origin for rotations,
        so best to use rotate() for that.

        :param matrix: the 2x2 transformation matrix
        """
        for curve in self:
            curve.apply_transformation_matrix(transformation_matrix)
        return self

    @property
    def num_atoms(self):
        num_atoms = 0
        for curve in self.curves:
            num_atoms += curve.positions.shape[0]
        return num_atoms

    @property
    def num_bonds(self):
        num_bonds = 0
        for curve in self.curves:
            num_bonds += len(curve.bonds)
        return num_bonds

    @property
    def num_angles(self):
        num_angles = 0
        for curve in self.curves:
            num_angles += len(curve.angles)
        return num_angles

    @property
    def positions(self):
        return np.concatenate([curve.positions for curve in self.curves], axis=0)

    def append(self, curve):
        """
        Add a new curve into our collection.
        :param curve: the curve to be added
        """
        self.curves.append(curve)

    def plot_onto(
        self, ax, kwarg_list=None, fit_edges: bool = True, label_nodes: bool =False,
    ):
        """
        Plots these polymers as a collection of lines, detailed by kwargs
        into the provided axis.
        :param ax: a matplotlib axis object
        :param fit_edges: should the axis size be rearranged to fit
        :param **kwargs: keyword arguments to be passed to matplotlib
        """

        if kwarg_list is None:
            kwarg_list = []

        length_difference = len(self) - len(kwarg_list)
        if length_difference > 0:
            kwarg_list.extend([{} for _ in range(length_difference)])
        elif length_difference < 0:
            print("More kwargs provided than there are curves." + "Ignoring excess")

        index_offset = 1
        for i, curve in enumerate(self.curves):
            # Don't pass on edge fitting because we'll
            # do it here.
            curve.plot_onto(ax, fit_edges=False, label_nodes=label_nodes, index_offset=index_offset, **kwarg_list[i])
            index_offset += curve.num_atoms

        if fit_edges:
            min_x, min_y = np.min(self.positions, axis=0)
            max_x, max_y = np.max(self.positions, axis=0)
            min_corner = (min(min_x, min_y) * 1.1) - 0.1
            max_corner = (max(max_x, max_y) * 1.1) + 0.1
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)
            
    def box_to_origin(self):
        min_x, min_y = np.min(self.positions, axis=0)
        self.translate(-np.array([min_x, min_y], dtype=float))

    def to_lammps(self, filename: str, periodic_box=None, mass=14.02):
        # Make the bottom left corner the origin
        self.box_to_origin()
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
            angle_types = set()
            for curve in self.curves:
                angle_types.update([angle[0] for angle in curve.angles])
            
            fi.write("\t " + f"{len(angle_types)}" + " \t angle types\n\n")

            if periodic_box is None:
                min_x, min_y = np.min(self.positions, axis=0)
                max_x, max_y = np.max(self.positions, axis=0)
                min_corner = 0.0
                max_corner = (max(max_x, max_y) * 1.1) + 0.1

                fi.write(f"\t {min_corner:.3f} {max_corner:.3f} \t xlo xhi\n")
                fi.write(f"\t {min_corner:.3f} {max_corner:.3f} \t ylo yhi\n")
                #
                fi.write(f"\t -{max_corner / 2} {max_corner / 2} \t zlo zhi\n\n")
            else:
                fi.write(
                    f"\t {periodic_box[0,0]:.3f} {periodic_box[0,1]:.3f} \t xlo xhi\n"
                )
                fi.write(
                    f"\t {periodic_box[1,0]:.3f} {periodic_box[1,1]:.3f} \t ylo yhi\n"
                )
                fi.write(f"\t -1.0 1.0 \t zlo zhi\n\n")
            # Masses
            fi.write("Masses\n\n")
            for atom_type in atom_types:
                fi.write(f"\t {atom_type} \t {mass}\n")
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
                    fi.write(
                        f"\t {atom_id} \t {molecule_id} \t {atom_type}"
                        + f"\t {atom[0]:.3f} \t {atom[1]:.3f} \t 0.000\n"
                    )
            fi.write("\n")

            # Bonds
            fi.write("Bonds\n\n")
            bond_id = 0
            total_atoms = 1
            for curve in self.curves:
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
            atom_offset = 1
            angle_id = 0
            for curve in self.curves:
                for angle in curve.angles:
                    angle_id += 1
                    # remember to fix a stupid lammps off-by-one
                    fi.write(
                        f"\t {angle_id} \t {angle[0]} \t {angle[1]  + atom_offset}"
                        + f"\t {angle[2] + atom_offset} \t {angle[3]  + atom_offset}\n"
                    )
                # Make sure we don't pile all the angles into one molecule
                atom_offset += curve.num_atoms
            fi.write("\n")
