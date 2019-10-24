#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:26:50 2019

@author: matthew-bailey
"""

from collections import defaultdict, Counter

import MDAnalysis as mda
import scipy.spatial.distance
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys



LJ_BOND = 1.5


def parse_molecule_topology(filename: str):
    """
    Extracts atom, molecule and position information
    from a LAMMPS data file.
    :param filename: the name of the lammps file to open
    :return atoms: a dictionary of atoms, with atom ids as keys
    and the values are a dictionary of the type and position.
    :return molecs: a dictionary of molecules, with molecule ids
    as keys and values a list of atoms in that molecule.
    :return bonds: a list of pairs, representing atom ids
    at each end of the bonds.
    """
    bonds_mode = False
    atoms_mode = False
    molecules = defaultdict(list)
    atoms = defaultdict(dict)
    bonds = []
    with open(filename, "r") as fi:
        for line in fi.readlines():
            if not line:
                continue
            if "Atoms" in line:
                atoms_mode = True
                bonds_mode = False
                continue
            if "Bonds" in line:
                atoms_mode = False
                bonds_mode = True
                continue
            if "Angles" in line:
                # To be implemented
                atoms_mode = False
                bonds_mode = False
                continue

            if atoms_mode:
                try:
                    atom_id, molec_id, atom_type, x, y, z = line.split()
                except ValueError:
                    print("Could not read line:", line,
                          "expected form: atom_id, molec_id, type, x, y, z")
                    continue
                atom_id = int(atom_id)
                molec_id = int(molec_id)
                atom_type = int(atom_type)
                x, y, z = float(x), float(y), float(z)
                atoms[atom_id] = {"type": atom_type,
                                  "pos":np.array([x, y, z])}
                molecules[molec_id].append(atom_id)
            if bonds_mode:
                try:
                    bond_id, bond_type, atom_a, atom_b = line.split()
                except ValueError:
                    print("Could not read bond line:", line,
                          "Expected form: bond_id, bond_type, a, b")
                    continue
                bonds.append([int(atom_a), int(atom_b)])
    return atoms, molecules, bonds


def find_lj_pairs(positions, ids, cutoff):
    """
    Calculate which pairs of atoms can be considered
    to have a Lennard-Jones bond between them,
    :param positions: an Nx3 numpy array of atomic positions
    :param cutoff: the distance below which they can be considered bonded
    :return lj_pairs: a dictionary of sets,
    """
    distances = scipy.spatial.distance.pdist(positions)
    distances = scipy.spatial.distance.squareform(distances)
    within_cutoff = np.argwhere(distances < cutoff)

    # Construct a dictionary that holds which pairs
    # are lj bonded. Does not currently take
    # periodic boundary conditions into account.
    lj_pairs = defaultdict(set)
    for atom_1, atom_2 in within_cutoff:
        if atom_1 != atom_2:
            atom_1_id, atom_2_id = ids[atom_1], ids[atom_2]
            lj_pairs[atom_1_id].add(atom_2_id)
            lj_pairs[atom_2_id].add(atom_1_id)

    return lj_pairs


def find_lj_clusters(pair_dict):
    """
    Find clusters of lennard jones pairs. Groups
    each set of Lennard-Jones pairs into a single set.
    :return clusters: a set of frozensets, with each
    entry containing a list of real molecules that make
    up a cluster.
    """
    clusters = set()
    for key, value in pair_dict.items():
        # This node should be in a cluster with not only
        # its neighbours, but its neighbours neighbours.
        second_degree_neighbours = [item 
                                    for neighbour in value
                                    for item in pair_dict[neighbour]]
        print(second_degree_neighbours)
        cluster = frozenset([key] + list(value) + second_degree_neighbours)
        print(value)
        clusters.add(cluster)
    return clusters


def find_cluster_centres(clusters, atom_positions):
    """
    Finds the centroid of all clusters of Lennard-Jones atoms.
    :param clusters: an ordered collection of frozensets, with each
    entry containing a list of real molecules that make
    up a cluster.
    :param atom_positions: a numpy array of atomic positions
    in order of id.
    :return cluster_positions: a dictionary of positions,
    with keys of atom ids and entries 2D positions.
    """
    cluster_positions = dict()
    # Make sure the clusters are ordered otherwise this will not
    # be helpful
    for i, cluster in enumerate(clusters):
        # LAMMPS indexes from 1, but positions doesn't!
        positions = [atom_positions[atom - 1] for atom in cluster]
        positions = np.vstack(positions)
        cluster_positions[i] = np.mean(positions, axis=0)[0:2]
    return cluster_positions


def find_molecule_terminals(molecules):
    """
    Finds the sticky ends of a LAMMPS molecule.
    Currently this is very dumb, and just takes the ends.
    :param mocecules: a dictionary of molecules, with molecule ids
    as keys and values a list of atoms in that molecule.
    :return molec_terminals: a dict, with keys being the ids of one
    terminal and the values being the ids of the other terminals
    in this molecule.
    """
    molec_terminals = dict()
    for molec in molecules.values():
        molec_terminals[molec[0]] = molec[-1]
        molec_terminals[molec[-1]] = molec[0]
    return molec_terminals


def connect_clusters(graph, terminals, clusters):
    """
    Connect the clusters to one another via the molecules
    that make them up.
    :param terminals:  a dict, with keys being the ids of one
    terminal and the values being the ids of the other terminals
    in this molecule.
    :param clusters: an ordered list of frozensets, with each
    entry containing a list of real molecules that make
    up a cluster.
    :param graph: an empty networkx graph to add these edges to.
    :return graph: a filled networkx graph with these edges in.
    """
    for i, cluster in enumerate(clusters):
        for atom in cluster:
            other_molec_end = terminals[atom]
            is_connected = np.array([other_molec_end in other_cluster
                                     for other_cluster in clusters])
            connected_clusters = [item for sublist in np.argwhere(is_connected)
                                  for item in sublist]
            for other_cluster in connected_clusters:
                graph.add_edge(i, other_cluster)
    return graph


if __name__ == "__main__":
    # Parsing section -- read the files, and split the atoms
    # and molecules into a few types. This could probably
    # be neatened with more use of MDA.
    if len(sys.argv) == 3:
        position_file = sys.argv[1]
        topology_file = sys.argv[2]
    else:
        position_file ="output_minimise.lammpstrj"
        topology_file = "polymer_total.data"
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
    MOLEC_TERMINALS = find_molecule_terminals(MOLECS)
    G = nx.Graph()
    G = connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
    with open("edges.dat", "w") as fi:
        fi.write("# Node A, Node B\n")
        for edge in G.edges():
            fi.write(f"{edge[0]}, {edge[1]}\n")
    with open("coords.dat", "w") as fi:
        fi.write("#Node ID, x , y\n")
        for node in sorted(G.nodes()):
            position = CLUSTER_POSITIONS[node]
            fi.write(f"{node}, {position[0]:.3f}, {position[1]:.3f}\n")

