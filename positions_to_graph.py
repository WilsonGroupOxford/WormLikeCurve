#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:26:50 2019

@author: matthew-bailey
"""

import sys
from collections import defaultdict, Counter

import MDAnalysis as mda
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from clustering import find_lj_pairs, find_lj_clusters, find_cluster_centres,\
find_molecule_terminals, connect_clusters
from lammps_parser import parse_molecule_topology
import rings

LJ_BOND = 1.5



            

if __name__ == "__main__":
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 3:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
    else:
        position_file = "./Data/output_minimise.lammpstrj"
        topology_file = "./Data/polymer_total.data"
    universe = mda.Universe(position_file,
                            topology=topology_file,
                            format="LAMMPSDUMP")
    ALL_ATOMS = universe.select_atoms("all")

    ATOMS, MOLECS, BONDS = parse_molecule_topology(topology_file)

    # Find the terminal atoms, and group them into clusters.
    TERMINALS = universe.select_atoms("type 2 or type 3")
    TERMINAL_PAIRS = find_lj_pairs(TERMINALS.positions,
                                   TERMINALS.ids,
                                   LJ_BOND)
    TERMINAL_CLUSTERS = find_lj_clusters(TERMINAL_PAIRS)

    # Sort the list of clusters into a consistent list so
    # we can index them.
    ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS))
    CLUSTER_POSITIONS = find_cluster_centres(ALL_CLUSTERS,
                                             ALL_ATOMS.positions)
    MOLEC_TERMINALS = find_molecule_terminals(MOLECS.values())
    G = nx.Graph()
    G = connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
    with open(position_file) as fi:
        to_read = 0
        box_data = []
        for line in fi.readlines():
            if to_read:
                box_data.append([float(item) for item in line.split()])
                to_read -= 1
            if "ITEM: BOX BOUNDS pp pp pp" in line:
                to_read = 3

    cell = np.array([[box_data[0][0], box_data[0][1]],
                     [box_data[1][0], box_data[1][1]]])

    coords = np.array([item for item in CLUSTER_POSITIONS.values()])
    target_network = rings.network.convert_from_networkx(G,
                                           coordinates=coords,
                                           cell_dim=cell,
                                           periodic=True)
    tri_network = rings.network.convert_from_networkx(G,
                                        coordinates=coords,
                                        cell_dim=cell,
                                        periodic=True)
    tri_network.triangulate()
    tri_network.map(target_network)
    
    ring_graph = nx.MultiDiGraph()
    for i, ring in enumerate(tri_network.rings):
        if ring.nodes:
            ring = rings.ring.reassemble_ring(G, ring.nodes, len(ring.nodes))
            G.add_node(i, ringsize=len(ring))
            for j, other_ring in enumerate(tri_network.rings):
                if i == j:
                    continue
                if other_ring.nodes:
                    other_ring = rings.ring.reassemble_ring(G,
                                                 other_ring.nodes,
                                                 len(other_ring.nodes))
                    shared_edges = rings.ring.calculate_shared_edges(ring,
                                                          other_ring)
                    if shared_edges:
                        print(shared_edges)
                        print(ring, other_ring, len(shared_edges))
                        ring_graph.add_edge(i, j, weight=len(shared_edges))

     
    ring_centres = dict()                  
    for i, ring in enumerate(tri_network.rings):
        if ring.nodes:
            node_positions = [CLUSTER_POSITIONS[node] for node in ring.nodes]
            node_positions = np.vstack(node_positions)
            ring_centres[i] = np.mean(node_positions, axis=0)
    fig, ax = plt.subplots()
    nx.draw(ring_graph, pos=ring_centres, ax=ax)
    fig.savefig("ring_graph.png")

    with open("./assortativity.dat", "w") as fi:
        assortativity = nx.attribute_assortativity_coefficient(ring_graph, "ringsize")
        fi.write(f"{assortativity}")

    with open("./coordination.dat", "w") as fi:
        coordination_counter = Counter([x[1] for x in G.degree])
        fi.write("Coordination, Frequency\n")
        for coord in sorted(coordination_counter.keys()):
            fi.write(f"{coord}, {coordination_counter[coord]}\n")

    with open("./ring_sizes.dat", "w") as fi:
        ring_sizes = Counter(len(ring.nodes) for ring in tri_network.rings)
        fi.write("Ring size, Frequency\n")
        for ring_size in sorted(ring_sizes.keys()):
            fi.write(f"{ring_size}, {ring_sizes[ring_size]}\n")
         
    plot_map = rings.plot_network.Plot(nodes=True,
                    cnxs=False,
                    rings=True,periodic=False)
    plot_map(tri_network, save=True, ms=20)
