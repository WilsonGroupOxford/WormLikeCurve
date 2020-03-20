#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:30:33 2019

@author: matthew-bailey
"""

import sys
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx

from WormLikeCurve.Bonds import AngleBond, HarmonicBond
from WormLikeCurve.CurveCollection import CurveCollection
from WormLikeCurve.WormLikeCurve import WormLikeCurve

KBT = 1.0  # atomic units
BOND_VAR = 1.0
MEAN_LENGTH = 6
STD_LENGTH = 0
DEFAULT_STICKY_FRACTION = 0.2
MIN_SIZE = 3

LINE_GRAPH = nx.path_graph(MEAN_LENGTH)
TRIANGLE_GRAPH = nx.Graph()
TRIANGLE_GRAPH.add_edges_from([(0, 1), (1, 2), (2, 3),
                               (0, 4), (4, 5), (5, 6),
                               (0, 7), (7, 8), (8, 9)])

DOUBLE_TRIANGLE_GRAPH = nx.Graph()
DOUBLE_TRIANGLE_GRAPH.add_edges_from([(0, 1), (1, 2), (2, 3),
                                      (0, 4), (4, 5), (5, 6),
                                      (0, 7), (7, 8), (8, 9),
                                      (9, 10), (10, 11), (11, 12),
                                      (9, 13), (13, 14), (14, 15)])
graphs = [LINE_GRAPH, TRIANGLE_GRAPH, DOUBLE_TRIANGLE_GRAPH]
CIRCUMCIRCLES = [WormLikeCurve(graph=graph).circumcircle_radius() for graph in graphs]
weights = [0.0, 1.0, 1.0]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        sticky_fraction = float(sys.argv[1])
    else:
        sticky_fraction = DEFAULT_STICKY_FRACTION

    if 0 > sticky_fraction:
        raise RuntimeError("Sticky fraction must be positive.")
    elif 1 < sticky_fraction:
        raise RuntimeError("Sticky fraction must be less than 1.")
    POLYMER_COLLECTION = CurveCollection()
    SPACING = 50 * 7
    num_x = 12
    num_y = 12
    positions_to_index = dict()
    num_added = 0
    for i in range(num_x):
        for j in range(num_y):
            # start_pos = np.array([x_offset, y_offset])
            start_pos = np.array([i * 2 * max(CIRCUMCIRCLES), j * 2 * max(CIRCUMCIRCLES)])
            # Iterate until we generate a positive size.
            size = -1
            while size < MIN_SIZE:
                size = int(np.random.normal(loc=MEAN_LENGTH, scale=STD_LENGTH))
            THIS_GRAPH = random.choices(graphs, weights=weights, k=1)[0]
            POLYMER_COLLECTION.append(
                WormLikeCurve(
                    graph=THIS_GRAPH,
                    harmonic_bond=HarmonicBond(k=1.0, length=50.0),
                    angle_bond=AngleBond(k=100.0, angle=np.pi),
                    start_pos=start_pos,
                )
            )
            POLYMER_COLLECTION[-1].recentre()
            # POLYMER_COLLECTION[-1].add_sticky_sites(sticky_fraction)
    FIG, AX = plt.subplots()
    AX.axis("equal")

    for POLYMER in POLYMER_COLLECTION:
        print(POLYMER.start_pos, POLYMER.centroid)
        POLYMER.rotate(np.random.uniform(0, 2 * np.pi))
        AX.add_artist(patches.Circle(POLYMER.centroid,  radius=POLYMER.circumcircle_radius(), edgecolor="black", fill=False))
    POLYMER_COLLECTION.plot_onto(AX)
    POLYMER_COLLECTION.to_lammps("polymer_total.data")
    plt.show()
