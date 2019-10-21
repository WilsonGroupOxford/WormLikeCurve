#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:16:27 2019

@author: matthew-bailey
"""



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
    # are lj bonded
    lj_pairs = defaultdict(set)
    for atom_1, atom_2 in within_cutoff:
        if atom_1 != atom_2:
            atom_1_id, atom_2_id = ids[atom_1], ids[atom_2]
            lj_pairs[atom_1_id].add(atom_2_id)
            lj_pairs[atom_2_id].add(atom_1_id)
    
    return lj_pairs

def find_lj_clusters(pair_dict):
    clusters = set()
    for key, value in pair_dict.items():
        cluster = tuple(sorted([key] + list(value)))
        clusters.add(cluster)
    return clusters

LJ_BOND = 1.5
if __name__ == "__main__":
    universe = mda.Universe("output_minimise.lammpstrj",
                        topology="polymer_total.data",
                        format="LAMMPSDUMP")
    type_2 = universe.select_atoms("type 2")
    type_3 = universe.select_atoms("type 3")
    # Calculate which type 3s are close to one another
    TYPE_2_PAIRS = find_lj_pairs(type_2.positions, type_2.ids, LJ_BOND)
    TYPE_3_PAIRS = find_lj_pairs(type_3.positions, type_3.ids, LJ_BOND)
    TYPE_3_CLUSTERS = find_lj_clusters(TYPE_3_PAIRS)
    print(TYPE_3_CLUSTERS)