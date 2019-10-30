#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:30:33 2019

@author: David Ormrod Morley
"""

import numpy as np
import rings
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":

    crds = np.genfromtxt("./coords.dat",
                         delimiter=",",
                         usecols=(1, 2)).astype(float)
    cnxs = np.genfromtxt("./edges.dat",
                         delimiter=",").astype(int)
    cell = np.genfromtxt("./cell.dat", delimiter=",")
    periodic = True
    target_network = rings.network.Network(crds,
                                   cell,
                                   periodic=periodic)
    target_network.construct(cnxs)

    tri_network = rings.network.Network(crds,
                          cell,
                          periodic=periodic)
    tri_network.construct(cnxs)
    tri_network.triangulate()

    plot_tri = rings.plot_network.Plot(nodes=True,
                    cnxs=False,
                    rings=True)
    plot_tri(tri_network,
             save=False,
             ms=20)

    plot_target = rings.plot_network.Plot(nodes=True,
                                          cnxs=True,
                                          rings=False)
    plot_target(target_network,
                save=False,
                ms=20)

    tri_network.map(target_network)
    plot_map = rings.plot_network.Plot(nodes=True,
                                       cnxs=False,
                                       rings=True,
                                       periodic=False)
    plot_map(tri_network,
             save=True,
             ms=20)

    coordinate_dict = {i: crds[i] for i in range(crds.shape[0])}
    FIG, AX = plt.subplots()
    G = rings.network.convert_to_networkx(tri_network)
    nx.draw(G, ax=AX, pos=coordinate_dict)
    FIG.show()
