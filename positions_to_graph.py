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

from clustering import (
    find_lj_pairs,
    find_lj_clusters,
    cluster_molecule_bodies,
    find_cluster_centres,
    find_molecule_terminals,
    connect_clusters,
)
from lammps_parser import parse_molecule_topology
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import RingFinder
from morley_parser import draw_periodic_coloured

LJ_BOND = 1.5
FIND_BODIES = False


class AnalysisFiles:
    def __init__(self, prefix: str):

        self.edges_prefixes = []
        self.edges_data = []
        self.edges_file = f"{prefix}_edges.dat"

        self.areas_prefixes = []
        self.areas_data = []
        self.areas_file = f"{prefix}_areas.dat"

        self.rings_prefixes = []
        self.rings_data = []
        self.rings_file = f"{prefix}_rings.dat"

        self.coordinations_prefixes = []
        self.coordinations_data = []
        self.coordinations_file = f"{prefix}_coordinations.dat"

    def write_coordinations(self, prefix, graph):
        """
        Buffer coordination data for later writing.

        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        coordination_counter = Counter([x[1] for x in graph.degree])
        self.coordinations_data.append(coordination_counter)
        self.coordinations_prefixes.append(str(prefix))

    def write_areas(self, prefix, ring_list):
        """
        Buffer sorted area data for later writing
        ----------
        ring_list : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        data = sorted([ring.area for ring in ring_list])
        self.areas_data.append(data)
        self.areas_prefixes.append(str(prefix))

    def write_edge_lengths(self, prefix, edge_length_list):
        self.edges_data.append(sorted(edge_length_list))
        self.edges_prefixes.append(str(prefix))

    def write_sizes(self, prefix, ring_list):
        ring_sizes = Counter(len(ring) for ring in ring_list)
        self.rings_data.append(ring_sizes)
        self.rings_prefixes.append(str(prefix))

    def flush(self):
        # Write out coordinations
        with open(self.coordinations_file, "w") as fi:
            all_coordinations = set()
            for row in self.coordinations_data:
                all_coordinations = all_coordinations.union(row.keys())
            all_coordinations = sorted(list(all_coordinations))
            fi.write(
                "# Timestep, "
                + ", ".join([str(coordination) for coordination in all_coordinations])
                + "\n"
            )
            for i, row in enumerate(self.coordinations_data):
                fi.write(
                    self.coordinations_prefixes[i]
                    + ",  "
                    + ", ".join(
                        [str(row[coordination]) for coordination in all_coordinations]
                    )
                    + "\n"
                )

        with open(self.areas_file, "w") as fi:
            fi.write("# Timestep, Ring Areas...\n")
            for i, row in enumerate(self.areas_data):
                fi.write(
                    self.areas_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )

        with open(self.rings_file, "w") as fi:
            all_sizes = set()
            for row in self.rings_data:
                all_sizes = all_sizes.union(row.keys())
            all_sizes = sorted(list(all_sizes))
            fi.write(
                "# Timestep, " + ", ".join([str(size) for size in all_sizes]) + "\n"
            )
            for i, row in enumerate(self.rings_data):
                fi.write(
                    self.rings_prefixes[i]
                    + ",  "
                    + ", ".join([str(row.get(size, 0)) for size in all_sizes])
                    + "\n"
                )

        with open(self.edges_file, "w") as fi:
            fi.write("# Timestep, Edge Lengths...\n")
            for i, row in enumerate(self.edges_data):
                fi.write(
                    self.edges_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )


if __name__ == "__main__":
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 4:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
        output_prefix = sys.argv[3]
    else:
        position_file = "./Data/FRZN_34_2.lammpstrj"
        topology_file = "./Data/FRZN_34_2.data"
        output_prefix = "./output/FRZN_34_2"

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
    ATOMS, MOLECS, BONDS = parse_molecule_topology(topology_file)
    ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
    MOLEC_TYPES = {
        molec_id: [ATOM_TYPES[atom_id] for atom_id in molec]
        for molec_id, molec in MOLECS.items()
    }
    OUTPUT_FILES = AnalysisFiles(output_prefix)
    for timestep in universe.trajectory[425::25]:
        # Find the terminal atoms, and group them into clusters.
        print(timestep)
        ALL_ATOMS = universe.select_atoms("all")
        ALL_ATOMS.positions *= 1.0 / np.array([x_size, y_size, 1.0])
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = find_lj_pairs(
            TERMINALS.positions, TERMINALS.ids, LJ_BOND, cell=cell
        )
        TERMINAL_CLUSTERS = find_lj_clusters(TERMINAL_PAIRS)

        if FIND_BODIES:
            BODIES = universe.select_atoms("type 4")
            BODY_PAIRS = find_lj_pairs(BODIES.positions, BODIES.ids, 1.5, cell=cell)
            body_molec_clusters = cluster_molecule_bodies(MOLECS, MOLEC_TYPES, [1, 4])
            for i, cluster in body_molec_clusters.items():
                BODY_PAIRS[i] = BODY_PAIRS[i].union(cluster)
            BODY_CLUSTERS = find_lj_clusters(BODY_PAIRS)
            ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
            TYPE_CONNECTIONS = {2: [1, 4], 3: [1, 4], 4: [2, 3], 1: [2, 3]}
        else:
            ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS))
            TYPE_CONNECTIONS = {2: [3], 3: [2]}
        # Sort the list of clusters into a consistent list so
        # we can index them.
        CLUSTER_POSITIONS = find_cluster_centres(
            ALL_CLUSTERS, ALL_ATOMS.positions, cutoff=10.0
        )

        MOLEC_TERMINALS = find_molecule_terminals(
            MOLECS, atom_types=MOLEC_TYPES, type_connections=TYPE_CONNECTIONS,
        )
        G = nx.Graph()
        try:
            G = connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
        except RuntimeError as ex:
            # We couldn't form any rings, so carry on merrily.
            print(ex)
            continue

        body_attr_dict = defaultdict(dict)
        terminal_attr_dict = defaultdict(dict)
        for CLUSTER_ID, CLUSTER in enumerate(ALL_CLUSTERS):
            if FIND_BODIES and CLUSTER in BODY_CLUSTERS:
                body_attr_dict[CLUSTER_ID]["type"] = "body"
            elif CLUSTER in TERMINAL_CLUSTERS:
                terminal_attr_dict[CLUSTER_ID]["type"] = "terminal"
            else:
                print(f"{CLUSTER} is not a valid cluster")
        nx.set_node_attributes(G, body_attr_dict)
        nx.set_node_attributes(G, terminal_attr_dict)
        colour_to_type = defaultdict(lambda: (2, 3))
        colour_to_type[0] = (2,)
        colour_to_type[1] = (3,)
        colours = nx.greedy_color(G, interchange=True)
        for key, value in colours.items():
            colours[key] = colour_to_type[value]
        nx.set_node_attributes(G, colours, name="color")
        # nx.write_edgelist(G, f"{position_file}_edges.dat", comments="#", delimiter=",")
        # with open(f"{position_file}_coords.dat", "w") as fi:
        #    fi.write("# ID, x, y\n")
        #    for key, value in sorted(CLUSTER_POSITIONS.items()):
        #        fi.write(f"{key}, {value[0]}, {value[1]}\n")
        FIG, AX = plt.subplots()
        try:
            ring_finder = PeriodicRingFinder(
                G, CLUSTER_POSITIONS, np.array([x_size, y_size])
            )
        except ValueError as ex:
            print(ex)
            plt.close(FIG)
            continue
        AX.set_xlim(box_data[0][0] * 1.1, box_data[0][1] * 1.1)
        AX.set_ylim(box_data[1][0] * 1.1, box_data[1][1] * 1.1)
        ring_finder.draw_onto(AX, cmap_name="tab20b", min_ring_size=4, max_ring_size=30)
        draw_periodic_coloured(
            G, pos=CLUSTER_POSITIONS, periodic_box=cell[:2, :], ax=AX
        )
        AX.axis("off")
        FIG.savefig(f"{output_prefix}_{universe.trajectory.time}.pdf")
        plt.close(FIG)

        OUTPUT_FILES.write_coordinations(universe.trajectory.time, G)
        OUTPUT_FILES.write_areas(universe.trajectory.time, ring_finder.current_rings)
        OUTPUT_FILES.write_sizes(universe.trajectory.time, ring_finder.current_rings)
        OUTPUT_FILES.write_edge_lengths(
            universe.trajectory.time, ring_finder.analyse_edges()
        )

    OUTPUT_FILES.flush()
