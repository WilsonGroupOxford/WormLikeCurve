#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:26:50 2019

@author: matthew-bailey
"""

import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
import PIL

from clustering import *
from lammps_parser import parse_molecule_topology
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import RingFinder

LJ_BOND = 1.5


if __name__ == "__main__":
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 3:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
    else:
        position_file = "./Data/stky_0.30_2.lammpstrj"
        topology_file = "./Data/stky_0.30_2.data"

    with open(position_file) as fi:
        to_read = 0
        box_data = []
        for line in fi.readlines():
            if to_read:
                box_data.append([float(item) for item in line.split()])
                to_read -= 1
            if "ITEM: BOX BOUNDS pp pp pp" in line:
                to_read = 3

    cell = np.array(
        [
            [box_data[0][0], box_data[0][1]],
            [box_data[1][0], box_data[1][1]],
            [box_data[2][0], box_data[2][1]],
        ]
    )
    x_size = abs(box_data[0][1] - box_data[0][0])
    y_size = abs(box_data[1][0] - box_data[1][1])

    universe = mda.Universe(position_file, topology=topology_file, format="LAMMPSDUMP")
    ALL_ATOMS = universe.select_atoms("all")

    ATOMS, MOLECS, BONDS = parse_molecule_topology(topology_file)
    ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
    MOLEC_TYPES = {
        molec_id: [ATOM_TYPES[atom_id] for atom_id in molec]
        for molec_id, molec in MOLECS.items()
    }

    # Find the terminal atoms, and group them into clusters.
    TERMINALS = universe.select_atoms("type 2 or type 3")
    TERMINAL_PAIRS = find_lj_pairs(
        TERMINALS.positions, TERMINALS.ids, LJ_BOND, cell=cell
    )
    TERMINAL_CLUSTERS = find_lj_clusters(TERMINAL_PAIRS)

    BODIES = universe.select_atoms("type 4")
    BODY_PAIRS = find_lj_pairs(BODIES.positions, BODIES.ids, 1.0, cell=cell)
    body_molec_clusters = cluster_molecule_bodies(MOLECS, MOLEC_TYPES, [1, 4])
    for i, cluster in body_molec_clusters.items():
        BODY_PAIRS[i] = BODY_PAIRS[i].union(cluster)
    BODY_CLUSTERS = find_lj_clusters(BODY_PAIRS)

    # Sort the list of clusters into a consistent list so
    # we can index them.
    ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
    CLUSTER_POSITIONS = find_cluster_centres(
        ALL_CLUSTERS, ALL_ATOMS.positions, cutoff=10.0
    )
    MOLEC_TERMINALS = find_molecule_terminals(
        MOLECS,
        atom_types=MOLEC_TYPES,
        type_connections={2: [1, 4], 3: [1, 4], 4: [2, 3], 1: [2, 3]},
    )
    G = nx.Graph()
    G = connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
    body_attr_dict = defaultdict(dict)
    terminal_attr_dict = defaultdict(dict)
    for CLUSTER_ID, CLUSTER in enumerate(ALL_CLUSTERS):
        if CLUSTER in BODY_CLUSTERS:
            body_attr_dict[CLUSTER_ID]["type"] = "body"
        elif CLUSTER in TERMINAL_CLUSTERS:
            terminal_attr_dict[CLUSTER_ID]["type"] = "terminal"
        else:
            print(f"{CLUSTER} is not a valid cluster")
    nx.set_node_attributes(G, body_attr_dict)
    nx.set_node_attributes(G, terminal_attr_dict)

    nx.write_edgelist(G, f"{position_file}_edges.dat", comments="#", delimiter=",")
    with open(f"{position_file}_coords.dat", "w") as fi:
        fi.write("# ID, x, y\n")
        for key, value in sorted(CLUSTER_POSITIONS.items()):
            fi.write(f"{key}, {value[0]}, {value[1]}\n")
    FIG, AX = plt.subplots()
    ring_finder = PeriodicRingFinder(G, CLUSTER_POSITIONS, np.array([x_size, y_size]))
    AX.set_xlim(-x_size * 0.5, x_size * 1.5)
    AX.set_ylim(-y_size * 0.5, y_size * 1.5)
    ring_finder.draw_onto(AX)
    # for perimeter_ring in ring_finder.perimeter_rings:
    #    edgelist = [tuple(item) for item in perimeter_ring.edges]
    #    nx.draw_networkx_edges(ring_finder.graph, ax=AX, pos=CLUSTER_POSITIONS,
    #                          edge_color="orange", zorder=1000, width=5,
    #                          edgelist=edgelist)
    FIG.savefig("./network.pdf")
    plt.close(FIG)

    with open("./edge_lengths.dat", "w") as fi:
        edge_lengths = ring_finder.analyse_edge_lengths()
        fi.write("\n".join([f"{length:.3f}" for length in edge_lengths]))

    with open("./terminal_angles.dat", "w") as fi:
        angles = ring_finder.analyse_node_angles(node_attrs="terminal")
        fi.write("\n".join([f"{angle:.3f}" for angle in angles]))

    with open("./body_angles.dat", "w") as fi:
        angles = ring_finder.analyse_node_angles(node_attrs="body")
        fi.write("\n".join([f"{angle:.3f}" for angle in angles]))

    with open("./coordination.dat", "w") as fi:
        coordination_counter = Counter([x[1] for x in G.degree])
        fi.write("Coordination, Frequency\n")
        for coord in sorted(coordination_counter.keys()):
            fi.write(f"{coord}, {coordination_counter[coord]}\n")

    with open("./areas.dat", "w") as fi:
        fi.write("# RingSize, Area\n")
        for ring in ring_finder.current_rings:
            fi.write(f"{len(ring)}, {ring.area}\n")

    with open("./ring_sizes.dat", "w") as fi:
        ring_sizes = Counter(len(ring) for ring in ring_finder.current_rings)
        fi.write("Ring size, Frequency\n")
        for ring_size in sorted(ring_sizes.keys()):
            fi.write(f"{ring_size}, {ring_sizes[ring_size]}\n")
