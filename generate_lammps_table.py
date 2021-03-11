#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:06:09 2021

@author: matthew-bailey
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

K_BOND = 0.16
R_EQM = 50.0
BURD_K_BOND = 1.5490e6 / 6 
R_STIFF = 1.347*R_EQM
SIGMA = 50.0
EPSILON = 4*4.142


def harm_exp_r(current_r):
    harm_term = (current_r-R_STIFF)**2
    stiff_eqm = (R_STIFF - R_EQM)**2
    exp_contents = -(current_r - R_EQM)/(R_STIFF-R_EQM)
    prefactor = BURD_K_BOND
    energy = prefactor * (harm_term + (stiff_eqm * (1.0 - (2.0*np.exp(exp_contents)))))
    energy[R_EQM > current_r] *= -1
    return energy


def harm_exp_force(current_r):
    harm_term = (current_r - R_STIFF)
    stiff_eqm = R_STIFF - R_EQM
    exp_contents = -(current_r - R_EQM)/(R_STIFF-R_EQM)
    prefactor = 2.0 * BURD_K_BOND
    force = prefactor * (harm_term + (stiff_eqm * np.exp(exp_contents)))
    force[R_EQM > current_r] *= -1
    return np.sign(current_r - R_STIFF) * force


def lj_12_4(current_r):
    print(SIGMA / current_r, SIGMA, current_r)
    return EPSILON * ((SIGMA / current_r)**12 - (SIGMA / current_r)**4)

def lj_12_4_force(current_r):
    return -(EPSILON/SIGMA) * (-12*(SIGMA / current_r)**13 + 4*(SIGMA / current_r)**5)
fig, ax = plt.subplots()

bond_xs = np.linspace(1, 100, 10001)
func_forces = harm_exp_force(bond_xs)
energies = harm_exp_r(bond_xs)

lj_xs = np.linspace(40, 200, 10001)
lj_energies = lj_12_4(lj_xs)
lj_forces = lj_12_4(lj_xs)

with open("./harm-exp.dat", "w") as fi:
    fi.write("# Harm-exp potential\n")
    fi.write("\n")
    fi.write("HARMEXP\n")
    fi.write(f"N {bond_xs.shape[0]} EQ {R_EQM}\n")
    fi.write("\n")
    for idx in range(bond_xs.shape[0]):
        fi.write(f"{idx+1} {bond_xs[idx]} {energies[idx]} {func_forces[idx]}\n")
        
with open("./lj-12-4.dat", "w") as fi:
    fi.write(f"# Lennard Jones 12-4 eps={EPSILON} sigma={SIGMA}\n")
    fi.write("\n")
    fi.write("LJ124\n")
    fi.write(f"N {lj_xs.shape[0]}\n")
    fi.write("\n")
    for idx in range(lj_xs.shape[0]):
        fi.write(f"{idx+1} {lj_xs[idx]} {lj_energies[idx]} {lj_forces[idx]}\n")
