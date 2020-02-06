#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:29:00 2019

@author: matthew-bailey
"""

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from WormLikeCurve.Bonds import HarmonicBond, AngleBond
from WormLikeCurve.Helpers import boltzmann_distrib

from WormLikeCurve.CurveCollection import CurveCollection


class WormLikeCurve:
    def __init__(self,
                 num_segments: int,
                 harmonic_bond: HarmonicBond,
                 angle_bond: AngleBond,
                 start_pos=np.array([0, 0]),
                 kbt: float = 1.0):
        """
        Initialise the polymer object.
        :param num_segments: the number of segments of the polymer
        :param harmonic_bond: a two-body bond object
        representing the atom-atom bonds
        :param angle_bond: a three-body bond object
        representing the atom-atom-atom angle potential
        :param kbt: the combination of boltzmann constant * temp
        in the same unit system as is used for the bonds.
        """
        self.num_segments = num_segments
        self.harmonic_bond = harmonic_bond
        self.angle_bond = angle_bond
        self.kbt = kbt
        self.vectors = np.empty([num_segments, 2])
        self.start_pos = start_pos
        self.bonds = []
        for i in range(num_segments):
            length = self.generate_length()
            angle = self.generate_angle(self.angle_bond,
                                        min_angle=0.0 * np.pi,
                                        max_angle=2.0 * np.pi)
            self.bonds.append((i, i+1))
            self.vectors[i, :] = length, angle
        self.atom_types = np.ones([self.num_atoms], dtype=int)
        self.atom_types[0] = 2
        self.atom_types[-1] = 3
        # Convert to Cartesian coordinates and recentre
        self.positions = self.vectors_to_positions()
        self.recentre()

    def add_sticky_sites(self, proportion: float):
        """
        Turns a proportion of sites along the body (type 1)
        into 'sticky' sites (type 4)
        """
        assert 0 < proportion, "Proportion must be a positive number between 0 and 1"
        assert 1 >= proportion, "Proportion must be a positive number between 0 and 1"
        num_body_atoms = self.num_atoms - 2
        body_atoms = np.random.choice([1, 4], p=[1.0-proportion, proportion], size=num_body_atoms)
        self.atom_types[1:-1] = body_atoms
        
    def recentre(self):
        """
        Recentres the polymer so that its centre of mass
        is at start_pos
        """
        self.positions -= self.centroid
        self.positions += self.start_pos

    def rotate(self, angle: float):
        """
        Rotates the polymer about its centroid.
        
        Does this
        by rotating the vector representation, and reconverting
        into Cartesians. Then recentres.
        :param angle: the angle to rotate about in radians.
        """
        self.vectors[:, 1] += angle
        self.positions = self.vectors_to_positions()
        self.recentre()

    @property
    def num_atoms(self):
        """
        Convenience property.
        
        Because bonds link two atoms, the
        number of atoms is one more than the number of bonds.
        """
        return self.num_segments + 1

    @property
    def num_bonds(self):
        return self.num_segments

    @property
    def num_angles(self):
        """
        Convenience property. Two bonds are linked by an angle,
        so the number of angles is one less than the number of bonds.
        """
        return self.num_segments - 1

    @property
    def centroid(self):
        """
        Returns the location of the centre of mass of this curve.
        """
        return np.mean(self.positions, axis=0)

    def generate_length(self,
                        min_length: float = None,
                        max_length: float = None,
                        max_iters: int = 10):
        """
        Generate a bond length for the harmonic bonds
        with a correct Boltzmann distribution.
        Randomly generates bond lengths between
        min_length and max_length, and calculates
        the energy. Makes a Monte Carlo-esque evaluation
        of the energy and rejects or accepts based on that.
        If the iterative process takes too long, just returns
        the mean length.
        :param min_length: minimum bond length, cannot be negative.
        Defaults to half the harmonic bond length if the argument is None
        :param max_length: minimum bond length, cannot be less than min_length.
        Defaults to twice the harmonic bond length if the argument is None.
        :param max_iters: how many iterations to try before returning the mean
        :return: length of a harmonic bond.
        """

        # Can't specify the lengths as an attribute in the signature.
        if max_length is None:
            max_length = self.harmonic_bond.length * 2

        if min_length is None:
            min_length = self.harmonic_bond.length / 2

        if min_length < 0:
            raise ValueError("Minimum length must be positive")

        if max_length < min_length:
            raise ValueError("Maximum length must be greater than or" +
                             "equal to minimum length")

        if max_iters < 0:
            raise ValueError("Maximum iterations must be a positive integer")
        iters_required = 0
        while True:
            iters_required += 1
            if iters_required > max_iters:
                return self.harmonic_bond.length
            length = np.random.uniform(min_length,
                                       max_length,
                                       1)
            probability = boltzmann_distrib(self.harmonic_bond.energy(length),
                                            self.kbt)
            if probability > np.random.uniform():
                return length

    def generate_angle(self,
                       angle_bond = None,
                       min_angle: float = 0,
                       max_angle: float = 2 * np.pi,
                       max_iters: int = 10):
        """
        Generate a bond angle for the angular
        with a correct Boltzmann distribution.
        Randomly generates bond angles between
        min_angle and max_angle, and calculates
        the energy. Makes a Monte Carlo-esque evaluation
        of the energy and rejects or accepts based on that.
        If the iterative process takes too long, just returns
        the mean length.
        :param min_angle: minimum bond angle, default is 0.
        :param max_angle: maximum bond angle, default is 2pi.
        :param max_iters: how many iterations to try before returning the mean
        :return: angle between this vector and the previous one.
        """
        if angle_bond is None:
            angle_bond = self.angle_bond
            
        if min_angle < 0 or min_angle > 2 * np.pi:
            raise ValueError("Minimum angle must be in the range" +
                             "0 to 2 pi.")

        if max_angle < min_angle:
            raise ValueError("Maximum angle must be greater than or" +
                             "equal to minimum angle")

        if max_angle < 0 or max_angle > 2 * np.pi:
            raise ValueError("Minimum angle must be in the range" +
                             "0 to 2 pi.")

        if max_iters < 0:
            raise ValueError("Maximum iterations must be a positive integer")

        iters_required = 0
        while True:
            iters_required += 1
            if iters_required > max_iters:
                return angle_bond.angle
            angle = np.random.uniform(min_angle,
                                      max_angle,
                                      1)

            probability = boltzmann_distrib(angle_bond.energy(angle),
                                            self.kbt)
            if probability > np.random.uniform():
                return angle

    def vectors_to_positions(self):
        """
        Converts the [r, theta] vectors
        into Cartesian coordinates.
        """
        positions = np.empty([self.num_segments + 1, 2])
        positions[0, :] = self.start_pos
        for i in range(self.num_segments):
            length, angle = self.vectors[i]
            vector = np.array([length * np.cos(angle),
                               length * np.sin(angle)])
            predecessor = self.bonds[i][0]
            positions[i + 1] = positions[predecessor] + vector
        return positions

    def plot_onto(self,
                  ax,
                  fit_edges: bool = True,
                  **kwargs):
        """
        Plots this polymer as a collection of lines, detailed by kwargs
        into the provided axis.
        :param ax: a matplotlib axis object to plot onto
        :param fit_edges: whether to resize xlim and ylim to fit this polymer.
        """

        lines = []
        for i in range(self.num_segments):
            predecessor = self.bonds[i][0]
            lines.append(np.row_stack([self.positions[predecessor],
                                       self.positions[i + 1]]))
        line_collection = LineCollection(lines, **kwargs)
        ax.add_collection(line_collection)

        # Now draw the sticky ends
        END_SIZE = 0.2
        for i in range(self.positions.shape[0]):
            if self.atom_types[i] == 1:
                color = "purple"
            elif self.atom_types[i] == 2:
                color = "blue"
            elif self.atom_types[i] == 3:
                color = "green"
            elif self.atom_types[i] == 4:
                color = "red"
            circle_end = mpatches.Circle(self.positions[i],
                                         END_SIZE / 2,
                                         color=color,
                                         **kwargs)
            ax.add_artist(circle_end)

        if fit_edges:
            min_x, min_y = np.min(self.positions, axis=0)
            max_x, max_y = np.max(self.positions, axis=0)
            min_corner = (min(min_x, min_y) * 1.1) - 0.1
            max_corner = (max(max_x, max_y) * 1.1) + 0.1
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)

    def to_lammps(self,
                  filename: str):
        """
        Writes out to a lammps file which can be read by
        a read_data command.
        :param filename: the name of the file to write to.
        """
        CurveCollection(self).to_lammps(filename)
