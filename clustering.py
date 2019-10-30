#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:06 2019

@author: matthew-bailey
"""

from collections import defaultdict
import numpy as np
import scipy.spatial.distance

def find_lj_pairs(positions, ids, cutoff):
    """
    Calculate which pairs of atoms can be considered
    to have a Lennard-Jones bond between them
    :param positions: an Nx3 numpy array of atomic positions
    :param ids: If these positions have come from a larger array, which
    what were their original indicies?
    :param cutoff: the distance below which they can be considered bonded
    :return lj_pairs: a dictionary of sets, each set representing
    all of the atoms in a cluster.
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
    
    # If an atom has no neighbours, the within_cutoff
    # array doesn't have an entry for it. Make sure
    # that we explicitly give this a pair of itself.     
    for atom_1 in range(positions.shape[0]):
        atom_1_id = ids[atom_1]
        lj_pairs[atom_1_id].add(atom_1_id)

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
        # and so on recursively.
        old_set = value
        while True:
            set_neighbours = set(item 
                              for neighbour in old_set
                              for item in pair_dict[neighbour])
            set_neighbours.update(old_set)
            if old_set != set_neighbours:
                old_set = set_neighbours
            else:
                break 
        clusters.add(frozenset(set_neighbours))
    return clusters


def find_cluster_centres(clusters, atom_positions, offset:int=1):
    """
    Finds the centroid of all clusters of Lennard-Jones atoms.
    :param clusters: an ordered collection of frozensets, with each
    entry containing a list of real molecules that make
    up a cluster.
    :param atom_positions: a numpy array of atomic positions
    in order of id.
    :param offset: how the ids are offset from positions.
    :return cluster_positions: a dictionary of positions,
    with keys of atom ids and entries 2D positions.
    """
    cluster_positions = dict()
    # Make sure the clusters are ordered otherwise this will not
    # be helpful
    for i, cluster in enumerate(clusters):
        # LAMMPS indexes from 1, but positions doesn't!
        positions = [atom_positions[atom - offset] for atom in cluster]
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
    for molec in molecules:
        try:
            molec_a = molec[0][0]
            molec_b = molec[-1][0]
        except TypeError:
            molec_a = molec[0]
            molec_b = molec[-1]
        molec_terminals[molec_a] = molec_b
        molec_terminals[molec_b] = molec_a
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
