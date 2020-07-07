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

KBT = 1e-10  # atomic units
BOND_VAR = 1.0
MEAN_LENGTH = 6
STD_LENGTH = 0
SEGMENT_LENGTH = 100.0
DEFAULT_STICKY_FRACTION = 0.0
MIN_SIZE = 3
NUM_X = 10
NUM_Y = 10

LINE_GRAPH = nx.path_graph(MEAN_LENGTH)
TRIANGLE_GRAPH = nx.Graph()
TRIANGLE_GRAPH.add_edges_from(
    [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9)]
)

DOUBLE_TRIANGLE_GRAPH = nx.Graph()
DOUBLE_TRIANGLE_GRAPH.add_edges_from(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
    ]
)
graphs = [LINE_GRAPH, TRIANGLE_GRAPH, DOUBLE_TRIANGLE_GRAPH]
CIRCUMCIRCLES = [
    WormLikeCurve(
        graph=graph, harmonic_bond=HarmonicBond(k=1.0, length=SEGMENT_LENGTH)
    ).circumcircle_radius()
    for graph in graphs
]
weights = [1.0, 0.5, 0.5]


def scale_rotate_to_fit(
    polymer_collection, iteration_scale: float = 0.1, rotation_size: float = 0.1 * np.pi
):
    """
    Find the minimum non-colliding size of box for these WLCs.
    
    This function aims to scale the centroid positions of the polymer_collection until we hit the
    minimum size of box that fits. Along the way, jiggle each WLC about by rotating it about its 
    centre of mass to see if that helps. 
    The collision detection is O(N^2), so be careful for large polymer collections!
    Mutates the polymer collection that is input.
    """
    any_colliders = True
    amount_rotated = np.zeros([len(polymer_collection)], dtype=float)

    # Reduce this from an O(N^2) horror each time by remembering which polys don't collide.
    # as they will continue not colliding until we move one of them.
    collider_list = np.ones(
        [len(polymer_collection), len(polymer_collection)], dtype=bool
    )
    while np.any(collider_list):
        for poly_idx, other_poly_idx in np.argwhere(collider_list):
            does_collide = polymer_collection[poly_idx].collides_with(
                polymer_collection[other_poly_idx]
            )
            if does_collide:
                # Two are colliding. Randomly rotate one of them by a small amount.
                poly_to_rotate = random.choice([poly_idx, other_poly_idx])
                polymer_collection[poly_to_rotate].rotate(rotation_size)
                amount_rotated[poly_to_rotate] += np.abs(rotation_size)

                # This could now collide with anything!
                collider_list[poly_to_rotate, :] = True
                collider_list[:, poly_to_rotate] = True
                # ... except itself, of course.
                collider_list[poly_to_rotate, poly_to_rotate] = False
            else:
                # These don't collide, so mark that down for future reference.
                collider_list[poly_idx, other_poly_idx] = False
                collider_list[other_poly_idx, poly_idx] = False

        # We've gone through a full rotation of one polymer and still not
        # fixed the problem. Scale up the whole lot.
        if np.any(amount_rotated > 2 * np.pi):
            print("Rescaling everything.")
            for poly in polymer_collection:
                poly.translate(poly.centroid * iteration_scale)
                poly.recentre()
            # Reset the rotations so we start again.
            amount_rotated = np.zeros([len(polymer_collection)], dtype=float)

        print(len(np.argwhere(collider_list)) / 2, " collisions to resolve.")
    return polymer_collection


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

    positions_to_index = dict()
    num_added = 0
    for i in range(NUM_X):
        for j in range(NUM_Y):
            # Iterate until we generate a positive size.
            size = -1
            while size < MIN_SIZE:
                size = int(np.random.normal(loc=MEAN_LENGTH, scale=STD_LENGTH))
            THIS_GRAPH = random.choices(graphs, weights=weights, k=1)[0]
            POLYMER_COLLECTION.append(
                WormLikeCurve(
                    graph=THIS_GRAPH,
                    harmonic_bond=HarmonicBond(k=1.0, length=SEGMENT_LENGTH),
                    angle_bond=AngleBond(k=100.0, angle=np.pi),
                )
            )
            POLYMER_COLLECTION[-1].add_sticky_sites(sticky_fraction)

    for i in range(NUM_X):
        for j in range(NUM_Y):
            new_start_pos = np.array(
                [i * 2 * min(CIRCUMCIRCLES), j * 2 * min(CIRCUMCIRCLES)]
            )
            idx = (i * NUM_X) + j
            POLYMER_COLLECTION[idx].translate(new_start_pos)
            POLYMER_COLLECTION[idx].recentre()

    scale_rotate_to_fit(POLYMER_COLLECTION)

    FIG, AX = plt.subplots()
    AX.axis("equal")

    for POLYMER in POLYMER_COLLECTION:
        POLYMER.rotate(np.random.uniform(0, 2 * np.pi))
        AX.add_artist(
            patches.Circle(
                POLYMER.centroid,
                radius=POLYMER.circumcircle_radius(),
                edgecolor="black",
                fill=False,
            )
        )
    POLYMER_COLLECTION.plot_onto(AX, label_nodes=False, fit_edges=True)
    print("Writing to polymer_total.data")
    POLYMER_COLLECTION.to_lammps("polymer_total.data", mass=0.5 / MEAN_LENGTH)
    plt.show()
