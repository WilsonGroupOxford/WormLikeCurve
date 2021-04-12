#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:29:32 2019

@author: matthew-bailey
"""

import numpy as np


class HarmonicBond:
    """
    Represents a LAMMPS harmonic bond between two atoms.
    """

    def __init__(self, k: float, length: float):
        """
        :param k: the force constant of the bond in a.u.
        :param length: the default length of the bond
        """
        self.k = k
        self.length = length
        if self.length < 0:
            raise ValueError(
                "Equilibrium length of the harmonic bond must" + "be positive."
            )

        if self.k < 0:
            raise ValueError("Force constant must be positive.")

    def energy(self, length: float) -> float:
        """
        Evaluates the energy of the harmonic bond
        in atomic units at a given length.
        :param length: the length of the bond currently
        :return: the energy at this length
        """
        return 0.5 * self.k * (length - self.length) ** 2


class AngleBond:
    """
    Represents a LAMMPS cosine/squared angle
    """

    def __init__(self, k: float, angle: float):
        """
        :param k: the force constant of the bond in a.u.
        :param angle: the equilibrium angle of the bond in radians
        """
        self.k = k
        self.angle = angle
        if abs(self.angle) > 2 * np.pi:
            print(
                "The angle is greater than 2pi. Did you mean to use"
                + "radians instead?"
            )
        self.cos_angle = np.cos(self.angle)

    def energy(self, angle: float) -> float:
        """
        Evaluates the energy of the angular bond
        in atomic units at a given angle
        :param angle: the angle of the bond currently
        :return: the energy at this angle
        """
        return self.k * (np.cos(angle) - self.cos_angle) ** 2
