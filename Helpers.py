#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:31:07 2019

@author: matthew-bailey
"""

import numpy as np


def boltzmann_distrib(energy: float, kbt: float) -> float:
    """
    Evaluates the probability of a particular energy occuring
    at a given thermal energy
    :param energy: the energy we want to see the likelihood of
    :param kbt: thermal energy available in atomic units
    :return: exp(-E/kbt), the probability of this energy being available.
    """
    return np.exp(-energy / kbt)
