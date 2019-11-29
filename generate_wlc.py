#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:30:33 2019

@author: matthew-bailey
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from WormLikeCurve.WormLikeCurve import WormLikeCurve
from WormLikeCurve.CurveCollection import CurveCollection
from WormLikeCurve.Bonds import HarmonicBond, AngleBond

KBT = 1.0  # atomic units
BOND_VAR = 1.0

if __name__ == "__main__":
    if len(sys.argv) == 2:
        sticky_fraction = float(sys.argv[1])
    POLYMER_COLLECTION = CurveCollection()
    SPACING = 7
    for i in range(12):
        for j in range(12):
            start_pos= np.array([-SPACING * i,  -SPACING * j])
            POLYMER_COLLECTION.append(WormLikeCurve(6,
                                                    HarmonicBond(k=1.0,
                                                                 length=1.0),
                                                    AngleBond(k=100.0,
                                                              angle=np.pi),
                                                    start_pos=start_pos
                                                    )
                                     )
            POLYMER_COLLECTION[-1].add_sticky_sites(sticky_fraction)
    FIG, AX = plt.subplots()
    AX.axis('equal')
    
    for POLYMER in POLYMER_COLLECTION:
        POLYMER.rotate(np.random.uniform(0, 2*np.pi))    
                                         
    
    POLYMER_COLLECTION.plot_onto(AX)
    POLYMER_COLLECTION.to_lammps("polymer_total.data")
    plt.show()
