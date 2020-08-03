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
    connect_clusters,
)
from lammps_parser import parse_molecule_topology
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import RingFinder, RingFinderError, convert_to_ring_graph
from morley_parser import draw_periodic_coloured
from nodeme import NodeME

LJ_BOND = 137.5
FIND_BODIES = True


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

        self.assortativity_prefixes = []
        self.assortativity_data = []
        self.assortativity_file = f"{prefix}_assortativity.dat"

        self.maxent_prefixes = []
        self.maxent_data = []
        self.maxent_file = f"{prefix}_maxent.dat"

        self.regularity_prefixes = []
        self.regularity_data = []
        self.regularity_file = f"{prefix}_regularity.dat"

    def write_coordinations(self, prefix: str, graph: nx.Graph):
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

    def write_areas(self, prefix: str, ring_list):
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

    def write_assortativity(self, prefix, assort):
        self.assortativity_data.append(str(assort))
        self.assortativity_prefixes.append(str(prefix))

    def write_sizes(self, prefix, ring_list):
        ring_sizes = Counter(len(ring) for ring in ring_list)
        self.rings_data.append(ring_sizes)
        self.rings_prefixes.append(str(prefix))

    def write_maximum_entropy(self, prefix, ring_list):
        ring_sizes = Counter(len(ring) for ring in ring_list)
        modal_ring_size, number_modal = ring_sizes.most_common(1)[0]
        me = NodeME(k_mean=modal_ring_size)(
            target_pk=number_modal / len(ring_list), k=modal_ring_size
        )
        print(me)
        self.maxent_data.append(me)
        self.maxent_prefixes.append(str(prefix))

    def write_regularity(self, prefix, ring_list):
        self.regularity_data.append([ring.regularity_metric() for ring in ring_list])
        self.regularity_prefixes.append(str(prefix))

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

        with open(self.assortativity_file, "w") as fi:
            fi.write("# Timestep, Ring Assortativity\n")
            for i, row in enumerate(self.assortativity_data):
                fi.write(self.assortativity_prefixes[i] + ",  " + row + "\n")

        with open(self.regularity_file, "w") as fi:
            fi.write("# Timestep, Regularities...\n")
            for i, row in enumerate(self.regularity_data):
                fi.write(
                    self.regularity_prefixes[i]
                    + ",  "
                    + ", ".join(f"{item:.2f}" for item in row)
                    + "\n"
                )

        with open(self.maxent_file, "w") as fi:
            fi.write(
                "# Timestep, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20\n"
            )
            for i, row in enumerate(self.maxent_data):
                fi.write(
                    self.maxent_prefixes[i]
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
        topology_file = "./Data/MSHP_0.05_2.data"
        output_prefix = "./outputs/MSHP_0.5_2"

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
    TOTAL_GRAPH = nx.Graph()
    TOTAL_GRAPH.add_edges_from(BONDS)
    ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
    nx.set_node_attributes(TOTAL_GRAPH, ATOM_TYPES, name="atom_types")
    MOLEC_TYPES = {
        molec_id: [ATOM_TYPES[atom_id] for atom_id in molec]
        for molec_id, molec in MOLECS.items()
    }
    OUTPUT_FILES = AnalysisFiles(output_prefix)
    for timestep in universe.trajectory[::25]:
        print(timestep)
        # Find the terminal atoms, and group them into clusters.
        ALL_ATOMS = universe.select_atoms("all")
        ALL_ATOMS.positions *= 1.0 / np.array([x_size, y_size, 1.0])
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = find_lj_pairs(
            TERMINALS.positions, TERMINALS.ids, LJ_BOND, cell=cell
        )
        TERMINAL_CLUSTERS = find_lj_clusters(TERMINAL_PAIRS)

        BODY_CLUSTERS = [
            frozenset([item]) for item in universe.select_atoms("type 4").ids
        ]
        ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
        # Sort the list of clusters into a consistent list so
        # we can index them.
        CLUSTER_POSITIONS = find_cluster_centres(
            ALL_CLUSTERS, ALL_ATOMS.positions, cutoff=50.0
        )
        G = connect_clusters(in_graph=TOTAL_GRAPH, clusters=ALL_CLUSTERS)
        colours = dict()
        for i, CLUSTER in enumerate(ALL_CLUSTERS):
            CLUSTER_ATOM_TYPES = [universe.atoms[atom - 1].type for atom in CLUSTER]
            MODAL_TYPE = Counter(CLUSTER_ATOM_TYPES).most_common(1)[0][0]
            colours[i] = (int(MODAL_TYPE),)
        # nx.draw(G, pos=CLUSTER_POSITIONS)
        nx.set_node_attributes(G, colours, name="color")
        FIG, AX = plt.subplots()
        AX.set_xlim(box_data[0][0] * 1.1, box_data[0][1] * 1.1)
        AX.set_ylim(box_data[1][0] * 1.1, box_data[1][1] * 1.1)
        RING_FINDER_SUCCESSFUL = True
        try:
            ring_finder = PeriodicRingFinder(
                G, CLUSTER_POSITIONS, np.array([x_size, y_size])
            )
            ring_finder.draw_onto(
                AX, cmap_name="tab20b", min_ring_size=4, max_ring_size=30
            )

            RING_GRAPH = convert_to_ring_graph(ring_finder.current_rings)
        except RingFinderError as ex:
            RING_FINDER_SUCCESSFUL = False
        except ValueError as ex:
            RING_FINDER_SUCCESSFUL = False

        draw_periodic_coloured(
            G, pos=CLUSTER_POSITIONS, periodic_box=cell[:2, :], ax=AX
        )

        AX.axis("off")
        FIG.savefig(f"{output_prefix}_{universe.trajectory.time}.pdf")
        plt.close(FIG)
        if RING_FINDER_SUCCESSFUL:

            OUTPUT_FILES.write_coordinations(universe.trajectory.time, G)
            OUTPUT_FILES.write_areas(
                universe.trajectory.time, ring_finder.current_rings
            )
            OUTPUT_FILES.write_sizes(
                universe.trajectory.time, ring_finder.current_rings
            )
            OUTPUT_FILES.write_regularities(
                universe.trajectory.time, ring_finder.current_rings
            )

            OUTPUT_FILES.write_maximum_entropy(
                universe.trajectory.time, ring_finder.current_rings
            )
            OUTPUT_FILES.write_edge_lengths(
                universe.trajectory.time, ring_finder.analyse_edges()
            )
            try:
                assortativity = nx.numeric_assortativity_coefficient(RING_GRAPH, "size")
            except ValueError:
                assortativity = np.nan
            OUTPUT_FILES.write_assortativity(universe.trajectory.time, assortativity)

    OUTPUT_FILES.flush()
