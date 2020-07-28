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
DEFAULT_STICKY_FRACTION = None
MIN_SIZE = 3
NUM_X = 20
NUM_Y = 20

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

GRAPHS = [LINE_GRAPH, TRIANGLE_GRAPH, DOUBLE_TRIANGLE_GRAPH]
DEFAULT_WEIGHTS = [1.0, 0.5, 0.5]

def get_neighbours_of(x, y, num_x, num_y):
    for offset_x, offset_y in [(-1, 1), (0, 1), (1, 1),
                               (-1, 0),         (1, 0),
                               (-1, -1),(0, -1), (1, -1)]:
        new_x = x + offset_x
        if new_x < 0:
            new_x += num_x
        elif new_x >= num_x:
            new_x -= num_x

        new_y = y + offset_y
        if new_y < 0:
            new_y += num_y
        elif new_y >= num_y:
            new_y -= num_y           
        yield new_x, new_y


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
    :param polymer_collection: A list of polymer objects
    :param iteration_scale: increase the linear dimensions by (1+iteration_scale) each scaling step
    :param rotation_size: rotate each polymer by rotation_size radians to resolve collisions. A smaller step resolves collisions more precisely, but is slower.
    """
    any_colliders = True
    amount_rotated = np.zeros([len(polymer_collection)], dtype=float)

    # Reduce this from an O(N^2) horror each time by remembering which polys don't collide.
    # as they will continue not colliding until we move one of them.
    collider_list = np.zeros(
        [len(polymer_collection), len(polymer_collection)], dtype=bool
    )
    
    # Polygons can only collide with their direct neighbours.
    # or, failing that, if they collide with other molecules we'll sort
    # that out when we sort out the neighbours.
    for x in range(NUM_X):
        for y in range(NUM_Y):
            idx = (x * NUM_X) + y
            for neigh_x, neigh_y in get_neighbours_of(x, y, NUM_X, NUM_Y):
                neigh_idx = (neigh_x * NUM_X) + neigh_y
                collider_list[idx, neigh_idx] = True
                collider_list[neigh_idx, idx] = True
    periodic_box = polymer_collection.calculate_periodic_box()
    num_iterations = 0
    while np.any(collider_list):
        print(f"After {num_iterations} iterations, there are {np.sum(collider_list) / 2} collisions to resolve.")
        for poly_idx, other_poly_idx in np.argwhere(collider_list):
            does_collide = polymer_collection[poly_idx].collides_with(
                polymer_collection[other_poly_idx], periodic_box = periodic_box,
            )
            if does_collide:
                # Two are colliding. Randomly rotate one of them by a small amount.
                poly_to_rotate = random.choice([poly_idx, other_poly_idx])
                polymer_collection[poly_to_rotate].rotate(rotation_size)
                amount_rotated[poly_to_rotate] += np.abs(rotation_size)

                # This could now collide with its neighbours, so find its x, y pair.
                poly_y = poly_to_rotate % NUM_Y
                poly_x = (poly_to_rotate - y) // NUM_X
                for neigh_x, neigh_y in get_neighbours_of(poly_x, poly_y, NUM_X, NUM_Y):
                    neigh_idx = (neigh_x * NUM_X) + neigh_y
                    collider_list[poly_to_rotate, neigh_idx] = True
                    collider_list[neigh_idx, poly_to_rotate] = True
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
            periodic_box = polymer_collection.calculate_periodic_box()

        num_iterations += 1
        
    return polymer_collection


if __name__ == "__main__":
    weights = DEFAULT_WEIGHTS
    sticky_fraction = DEFAULT_STICKY_FRACTION
    if len(sys.argv) == 2:
        sticky_fraction = float(sys.argv[1])      
    elif len(sys.argv) == 4:
        weights = [float(item) for item in sys.argv[1:]]

    POLYMER_COLLECTION = CurveCollection()

    positions_to_index = dict()
    num_added = 0
    NUM_MOLECS = NUM_X * NUM_Y
    # Normalise the weights such that they sum to unity.
    weights = np.array(weights) / np.sum(weights)
    CIRCUMCIRCLES = []
    for i, weight in enumerate(weights):
        # print(weight, weight * NUM_MOLECS, int(weight * NUM_MOLECS))
        for poly in range(int(weight * NUM_MOLECS)):
            POLYMER_COLLECTION.append(
                WormLikeCurve(
                    graph=GRAPHS[i],
                    harmonic_bond=HarmonicBond(k=1.0, length=SEGMENT_LENGTH),
                    angle_bond=AngleBond(k=100.0, angle=np.pi),
                    sticky_fraction=sticky_fraction,
                )
            )
            CIRCUMCIRCLES.append(POLYMER_COLLECTION[-1].circumcircle_radius())
    
    # If we're short, pad with the most populous subtype.
    while len(POLYMER_COLLECTION) < NUM_MOLECS:
        most_populous = np.argmax(weights)
        POLYMER_COLLECTION.append(
            WormLikeCurve(
                graph=GRAPHS[most_populous],
                harmonic_bond=HarmonicBond(k=1.0, length=SEGMENT_LENGTH),
                angle_bond=AngleBond(k=100.0, angle=np.pi),
                sticky_fraction=sticky_fraction,
            )
        )
        CIRCUMCIRCLES.append(POLYMER_COLLECTION[-1].circumcircle_radius())
    
    # Remove any polymers that are in excess.
    while len(POLYMER_COLLECTION) > NUM_MOLECS:
        del POLYMER_COLLECTION[-1]
        
    random.shuffle(POLYMER_COLLECTION)
    
    for i in range(NUM_X):
        for j in range(NUM_Y):
            new_start_pos = np.array(
                [i * 2 * min(CIRCUMCIRCLES), j * 2 * min(CIRCUMCIRCLES)]
            )
            idx = (i * NUM_X) + j
            POLYMER_COLLECTION[idx].translate(new_start_pos)
            POLYMER_COLLECTION[idx].recentre()

    scale_rotate_to_fit(POLYMER_COLLECTION)

    #FIG, AX = plt.subplots()
    #AX.axis("equal")

    for POLYMER in POLYMER_COLLECTION:
        POLYMER.rotate(np.random.uniform(0, 2 * np.pi))
    #    AX.add_artist(
    #        patches.Circle(
    #            POLYMER.centroid,
    #            radius=POLYMER.circumcircle_radius(),
    #            edgecolor="black",
    #            fill=False,
    #        )
    #    )
    #POLYMER_COLLECTION.plot_onto(AX, label_nodes=False, fit_edges=True)
    print("Writing to polymer_total.data")
    POLYMER_COLLECTION.to_lammps("polymer_total.data", mass=0.5 / MEAN_LENGTH)
    #FIG.savefig("./initial.pdf")
    # plt.show()
