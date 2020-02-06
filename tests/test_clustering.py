#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:06 2019

@author: matthew-bailey

A set of tests for turning LAMMPS outputs into 
graphs, which involves clustering the Lennard-Jones
particles into single graph nodes and connecting them.
"""

from collections import Counter

import pytest
import clustering
import lammps_parser
import rings.periodic_ring_finder
import numpy as np
import MDAnalysis as mda
import networkx as nx
import matplotlib.pyplot as plt

def generate_polygon(num_sides: int, radius:float = 1.0, centre=np.zeros(2),
                     edge_offset = 0):
    angle: float = 2 * np.pi / num_sides
    coordinates = dict()
    for i in range(num_sides):
        this_coord = np.array((radius * np.sin(i * angle), 
                               radius * np.cos(i * angle)))
        coordinates[i] = this_coord + centre
    return coordinates

class TestShapes:
    def test_find_octagon_pairs(self):
        """
        Make sure we find all the vertices of an octagon as being paired
        to the next two vertices.
        """
        coords = generate_polygon(8, 1.0)
        coords = np.array([value for value in coords.values()])
        ids = np.array([i for i in range(len(coords))])
        pairs = clustering.find_lj_pairs(positions=coords, ids=ids, cutoff=1.01)
        # We've found one set of pairs per node.
        assert len(pairs) == 8
        for key, value in pairs.items():
            # Check that (n-1), (n) and (n+1) are the pairs.
            assert len(value) == 3
            assert key in value
            assert (key + 1) % 8 in value
            if (key - 1) < 0:
                assert key + 7 in value
            else:
                assert key - 1 in value

    def test_find_octagon_clusters(self):
        """
        Make sure we cluster all the vertices of one octagon
        into a single point.
        """
        coords = generate_polygon(8, 1.0)
        coords = np.array([value for value in coords.values()])
        ids = np.array([i for i in range(len(coords))])
        pairs = clustering.find_lj_pairs(positions=coords, ids=ids, cutoff=1.01)
        clusters = clustering.find_lj_clusters(pairs)
        assert len(clusters) == 1
        cluster = clusters.pop()
        # Check everything we expect is in there
        for i in range(8):
            assert i in cluster
        # and check there's nothing we don't expect.
        for item in cluster:
            assert item in set(range(8))

    def test_find_octagon_centre(self):
        """
        Make sure the centre of the octagon is where
        we thought it was.
        """
        coords = generate_polygon(8, 1.0)
        coords = np.array([value for value in coords.values()])
        ids = np.array([i for i in range(len(coords))])
        pairs = clustering.find_lj_pairs(positions=coords, ids=ids, cutoff=1.01)
        clusters = clustering.find_lj_clusters(pairs)
        centres = clustering.find_cluster_centres(atom_positions=coords, clusters=clusters, offset=0)
        assert len(centres) == 1
        centre = centres[0]
        assert np.all(np.abs(centre) < 1e-16)

class TestPeriodicBoundary:
    """
    Tests to ensure that a cluster spanning periodic boundary
    conditions is treated sanely. In an infinitely large box,
    these will never occur. Sadly, finite boxes are not so
    easy and there is a small but non-zero chance a cluster
    spans a box edge.
    """
    def test_minimum_image(self):
        """
        Add two points at opposite sides of a cell, and see
        if we correctly cluster them.
        """
        coords = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]])
        pairs = clustering.find_lj_pairs(positions=coords,
                                         ids=np.array([0, 1]),
                                         cutoff=0.5,
                                         cell=np.array([[0.0, 1.0],
                                                        [0.0, 1.0],
                                                        [0.0, 1.0]]))
        assert len(pairs) == 2
        for item, value in pairs.items():
            assert value == set([0, 1])

    def test_minimum_diagonal(self):
        """
        Add two points at diagonal sides of a cell and see if
        we correctly cluster them.
        """
        coords = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        pairs = clustering.find_lj_pairs(positions=coords,
                                         ids=np.array([0, 1]),
                                         cutoff=0.5,
                                         cell=np.array([[0.0, 1.0],
                                                        [0.0, 1.0],
                                                        [0.0, 1.0]]))
        assert len(pairs) == 2
        for item, value in pairs.items():
            assert value == set([0, 1])

    def test_minimum_diagonal_cube(self):
        """
        Add points at each corner of the cell and cluster them. 
        Make sure we can get the cluster to include all of
        the corners.
        """
        coords = np.array([[0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.9],
                           [0.1, 0.9, 0.1],
                           [0.9, 0.1, 0.1],
                           [0.1, 0.9, 0.9],
                           [0.9, 0.1, 0.9],
                           [0.9, 0.9, 0.1],
                           [0.9, 0.9, 0.9]])
        pairs = clustering.find_lj_pairs(positions=coords,
                                         ids=np.array(range(8)),
                                         cutoff=0.5,
                                         cell=np.array([[0.0, 1.0],
                                                        [0.0, 1.0],
                                                        [0.0, 1.0]]))
        assert len(pairs) == 8
        for item, value in pairs.items():
            assert value == set(range(8))

    def test_minimum_diagonal_centre(self):
        """
        Add points at each corner of the cell and cluster them.
        Make sure we don't end up with the centre of this cluster
        being at [0.5, 0.5, 0.5]!
        """
        coords = np.array([[0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.9],
                           [0.1, 0.9, 0.1],
                           [0.9, 0.1, 0.1],
                           [0.1, 0.9, 0.9],
                           [0.9, 0.1, 0.9],
                           [0.9, 0.9, 0.1],
                           [0.9, 0.9, 0.9]])
        pairs = clustering.find_lj_pairs(positions=coords,
                                         ids=np.array(range(8)),
                                         cutoff=0.5,
                                         cell=np.array([[0.0, 1.0],
                                                        [0.0, 1.0],
                                                        [0.0, 1.0]]))
        clusters = clustering.find_lj_clusters(pairs)
        assert len(clusters) == 1

        centre = clustering.find_cluster_centres(clusters, coords, cutoff=1.0)
        assert len(centre) == 1
        assert centre[0][0] == 0.9
        assert centre[0][1] == 0.9
        # assert centre[0][2] == 0.9

    def test_multi_type(self):
        """ 
        Check that we correctly cluster body groups and end groups
        without any crossover
        """
        atom_types = np.array([2, 4, 3, 2, 4, 3], dtype=int)
        positions = np.array([[ 0.0, 0.0, 0.0],
                              [ 0.0, 1.0, 0.0],
                              [-1.0, 1.0, 0.0],
                              [ 1.0, 0.0, 0.0],
                              [ 1.0, 1.0, 0.0],
                              [ 2.0, 1.0, 0.0]])
        molecs = {0: [0, 1, 2], 1: [3, 4, 5]}
        molec_types = {molec_id: [atom_types[atom_id] for atom_id in molec]
                       for molec_id, molec in molecs.items()}
        type_4_ids = np.argwhere(atom_types == 4)[:, 0]
        print(type_4_ids.shape)
        type_2_3_ids = np.argwhere(np.logical_or(atom_types == 2, atom_types == 3))[:, 0]
        terminal_pairs = clustering.find_lj_pairs(positions[type_2_3_ids],
                                                  type_2_3_ids,
                                                  cutoff=1.1)
        terminal_clusters = clustering.find_lj_clusters(terminal_pairs)

        body_pairs = clustering.find_lj_pairs(positions[type_4_ids],
                                              type_4_ids,
                                              cutoff=1.1)
        body_clusters = clustering.find_lj_clusters(body_pairs)
        assert terminal_clusters == set([frozenset([0, 3]), frozenset([2]), frozenset([5])])
        assert body_clusters == set([frozenset([1, 4])])



class TestStickySites:
    """
    Tests for adding sticky body sites to the chains
    """
    def test_multi_type_connection(self):
        """ 
        Check that we correctly cluster body groups and end groups
        without any crossover and then correctly connect them.
        """
        atom_types = np.array([2, 4, 3, 2, 4, 3], dtype=int)
        positions = np.array([[ 0.0, 0.0, 0.0],
                              [ 0.0, 1.0, 0.0],
                              [-1.0, 1.0, 0.0],
                              [ 1.0, 0.0, 0.0],
                              [ 1.0, 1.0, 0.0],
                              [ 2.0, 1.0, 0.0]])
        molecs = {0: [0, 1, 2], 1: [3, 4, 5]}
        molec_types = {molec_id: [atom_types[atom_id] for atom_id in molec]
                       for molec_id, molec in molecs.items()}
        type_4_ids = np.argwhere(atom_types == 4)[:, 0]
        type_2_3_ids = np.argwhere(np.logical_or(atom_types == 2, atom_types == 3))[:, 0]
        terminal_pairs = clustering.find_lj_pairs(positions[type_2_3_ids],
                                                  type_2_3_ids,
                                                  cutoff=1.1)
        terminal_clusters = clustering.find_lj_clusters(terminal_pairs)

        body_pairs = clustering.find_lj_pairs(positions[type_4_ids],
                                              type_4_ids,
                                              cutoff=1.1)
        body_clusters = clustering.find_lj_clusters(body_pairs)
        G = nx.Graph()
        terminals = clustering.find_molecule_terminals(molecs,
                                                    molec_types,
                                                    type_connections={2:[4], 3:[4], 4:[2, 3]})
        clusters = body_clusters.union(terminal_clusters)
        assert terminals == {0:[1], 1:[0, 2], 2:[1], 3:[4], 4:[3, 5], 5:[4]}
        
        G = nx.Graph()
        G = clustering.connect_clusters(G, terminals, clusters)
        # We expect a T-shaped structure from this.
        assert len(G.edges) == 3
        assert list(G.edges) == [(0, 2), (2, 1), (2, 3)]

    def test_cluster_molecule_bodies(self):
        positions = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [2.0, 0.0, 0.0],
                              [3.0, 0.0, 0.0],
                              [4.0, 0.0, 0.0],
                              [5.0, 0.0, 0.0],
                              [6.0, 0.0, 0.0]])
        atom_types = np.array([2, 4, 1, 1,  1, 4, 3])
        molecs = {0: [0, 1, 2, 3, 4, 5, 6]}
        molec_types = {molec_id: [atom_types[atom_id] for atom_id in molec]
                       for molec_id, molec in molecs.items()}
        type_4_ids = np.argwhere(atom_types == 4)[:, 0]
        type_2_3_ids = np.argwhere(np.logical_or(atom_types == 2, atom_types == 3))[:, 0]

        terminal_pairs = clustering.find_lj_pairs(positions[type_2_3_ids],
                                                  type_2_3_ids,
                                                  cutoff=1.1)
        terminal_clusters = clustering.find_lj_clusters(terminal_pairs)

        body_pairs = clustering.find_lj_pairs(positions[type_4_ids],
                                              type_4_ids,
                                              cutoff=1.1)
        body_molec_clusters = clustering.cluster_molecule_bodies(molecs, molec_types, [4])
        assert body_molec_clusters == {1: {1, 5}, 5: {5, 1}}
        for i, cluster in body_molec_clusters.items():
            body_pairs[i] = body_pairs[i].union(cluster)
        assert body_molec_clusters == {1: {1, 5}, 5: {5, 1}}
        body_clusters = clustering.find_lj_clusters(body_pairs)
        assert body_clusters == {frozenset([1, 5])}

class TestIntegration:
    """
    Full clustering integration tests on real data.
    """
    @pytest.mark.slow
    def test_coll_2_3(self):
        """
        Does a minimum image convention clustering analysis on real data,
        but takes about 15s.
        """
        position_file = "./Data/test_collagen_clusters.lammpstrj"
        topology_file = "./Data/test_collagen_clusters.data"
        cell = np.array([[-1.4746951924029645e+02, 1.3145192404711850e+00],
                         [-1.4746951924029645e+02, 1.3145192404711850e+00],
                         [-1.0, 1.0]])
        universe = mda.Universe(position_file,
                                topology=topology_file,
                                format="LAMMPSDUMP")
        ALL_ATOMS = universe.select_atoms("all")

        ATOMS, MOLECS, BONDS = lammps_parser.parse_molecule_topology(topology_file)

        # Find the terminal atoms, and group them into clusters.
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = clustering.find_lj_pairs(TERMINALS.positions,
                                                  TERMINALS.ids,
                                                  cutoff=1.6,
                                                  cell=cell)
        TERMINAL_CLUSTERS = clustering.find_lj_clusters(TERMINAL_PAIRS)
        cluster_sizes = Counter([len(cluster) for cluster in TERMINAL_CLUSTERS])
        correct_count = {1:26, 2:39, 3:224, 4:6}
        for key, value in cluster_sizes.items():
            assert cluster_sizes[key] == correct_count[key]

    @pytest.mark.slow
    def test_wrong_edge_arguments(self):
        """
        An older version of this program crashed because it provided the 
        wrong number of arguments sometimes in the edge flipping routine.
        Let's make sure that it fails in a consistent way.
        """
        position_file = "./Data/test_wrong_edge_arguments.lammpstrj"
        topology_file = "./Data/test_wrong_edge_arguments.data"


        cell = np.array([[-6.6911374900124301e+01, -1.7635625099875906e+01],
                         [-6.6911374900124301e+01, -1.7635625099875906e+01],
                         [-1.0, 1.0]])
        x_size = abs(cell[0, 1] - cell[0, 0])
        y_size = abs(cell[1, 1] - cell[1, 0])


        universe = mda.Universe(position_file,
                                topology=topology_file,
                                format="LAMMPSDUMP")
        ALL_ATOMS = universe.select_atoms("all")

        ATOMS, MOLECS, BONDS = lammps_parser.parse_molecule_topology(topology_file)
        ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
        MOLEC_TYPES = {molec_id: [ATOM_TYPES[atom_id] for atom_id in molec] for molec_id, molec in MOLECS.items()}

        # Find the terminal atoms, and group them into clusters.
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = clustering.find_lj_pairs(TERMINALS.positions,
                                       TERMINALS.ids,
                                       1.5,
                                       cell=cell)
        TERMINAL_CLUSTERS = clustering.find_lj_clusters(TERMINAL_PAIRS)

        BODIES = universe.select_atoms("type 4")
        BODY_PAIRS = clustering.find_lj_pairs(BODIES.positions,
                                   BODIES.ids,
                                   1.0,
                                   cell=cell)
        body_molec_clusters = clustering.cluster_molecule_bodies(MOLECS, MOLEC_TYPES, [1, 4])
        for i, cluster in body_molec_clusters.items():
            BODY_PAIRS[i] = BODY_PAIRS[i].union(cluster)
        BODY_CLUSTERS = clustering.find_lj_clusters(BODY_PAIRS)

        # Sort the list of clusters into a consistent list so
        # we can index them.
        ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
        CLUSTER_POSITIONS = clustering.find_cluster_centres(ALL_CLUSTERS,
                                                 ALL_ATOMS.positions,
                                                 cutoff=10.0)
        MOLEC_TERMINALS = clustering.find_molecule_terminals(MOLECS,
                                                             atom_types=MOLEC_TYPES,
                                                             type_connections={2:[1, 4],      
                                                                               3:[1, 4],
                                                                               4:[2,3],
                                                                               1:[2, 3]})
        G = nx.Graph()
        G = clustering.connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
        FIG, AX = plt.subplots()
        nx.draw(G, ax=AX, pos=CLUSTER_POSITIONS, node_size=3)
        FIG.savefig("./test_wrong_arguments.pdf", dpi=800)
        with pytest.raises(RuntimeError):
            ring_finder = rings.periodic_ring_finder.PeriodicRingFinder(G, CLUSTER_POSITIONS, np.array([x_size, y_size]))


    def test_doublecount_large(self):
        """
        An older version of this program crashed because it provided the 
        wrong number of arguments sometimes in the edge flipping routine.
        Let's make sure that it fails in a consistent way.
        """
        position_file = "./Data/test_doublecount_large.lammpstrj"
        topology_file = "./Data/test_doublecount_large.data"


        cell = np.array([[-8.4163209853942732e+01, -1.0617901460540153e+00],
                         [-8.4163209853942732e+01, -1.0617901460540153e+00],
                         [-1.0, 1.0]])
        x_size = abs(cell[0, 1] - cell[0, 0])
        y_size = abs(cell[1, 1] - cell[1, 0])


        universe = mda.Universe(position_file,
                                topology=topology_file,
                                format="LAMMPSDUMP")
        ALL_ATOMS = universe.select_atoms("all")

        ATOMS, MOLECS, BONDS = lammps_parser.parse_molecule_topology(topology_file)
        ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
        MOLEC_TYPES = {molec_id: [ATOM_TYPES[atom_id] for atom_id in molec] for molec_id, molec in MOLECS.items()}

        # Find the terminal atoms, and group them into clusters.
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = clustering.find_lj_pairs(TERMINALS.positions,
                                       TERMINALS.ids,
                                       1.5,
                                       cell=cell)
        TERMINAL_CLUSTERS = clustering.find_lj_clusters(TERMINAL_PAIRS)

        BODIES = universe.select_atoms("type 4")
        BODY_PAIRS = clustering.find_lj_pairs(BODIES.positions,
                                   BODIES.ids,
                                   1.0,
                                   cell=cell)
        body_molec_clusters = clustering.cluster_molecule_bodies(MOLECS, MOLEC_TYPES, [1, 4])
        for i, cluster in body_molec_clusters.items():
            BODY_PAIRS[i] = BODY_PAIRS[i].union(cluster)
        BODY_CLUSTERS = clustering.find_lj_clusters(BODY_PAIRS)

        # Sort the list of clusters into a consistent list so
        # we can index them.
        ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
        CLUSTER_POSITIONS = clustering.find_cluster_centres(ALL_CLUSTERS,
                                                 ALL_ATOMS.positions,
                                                 cutoff=10.0)
        MOLEC_TERMINALS = clustering.find_molecule_terminals(MOLECS,
                                                             atom_types=MOLEC_TYPES,
                                                             type_connections={2:[1, 4],      
                                                                               3:[1, 4],
                                                                               4:[2,3],
                                                                               1:[2, 3]})
        G = nx.Graph()
        G = clustering.connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
        ring_finder = rings.periodic_ring_finder.PeriodicRingFinder(G, CLUSTER_POSITIONS, np.array([x_size, y_size]))
        counts = Counter([len(ring) for ring in ring_finder.current_rings])
        true_counts = {8: 6, 12: 5, 48: 2, 30: 2, 28: 1, 80: 1, 20: 1}
        assert counts == true_counts

    def test_clustering_error(self):
        position_file = "./Data/test_clustering_error.lammpstrj"
        topology_file = "./Data/test_clustering_error.data"


        cell = np.array([[-7.7994171576760237e+01, -6.4008284232299033e+00],
                         [-7.7994171576760237e+01, -6.4008284232299033e+00],
                         [-1.0, 1.0]])
        x_size = abs(cell[0, 1] - cell[0, 0])
        y_size = abs(cell[1, 1] - cell[1, 0])


        universe = mda.Universe(position_file,
                                topology=topology_file,
                                format="LAMMPSDUMP")
        ALL_ATOMS = universe.select_atoms("all")

        ATOMS, MOLECS, BONDS = lammps_parser.parse_molecule_topology(topology_file)
        ATOM_TYPES = {atom_id: atom["type"] for atom_id, atom in ATOMS.items()}
        MOLEC_TYPES = {molec_id: [ATOM_TYPES[atom_id] for atom_id in molec] for molec_id, molec in MOLECS.items()}

        # Find the terminal atoms, and group them into clusters.
        TERMINALS = universe.select_atoms("type 2 or type 3")
        TERMINAL_PAIRS = clustering.find_lj_pairs(TERMINALS.positions,
                                       TERMINALS.ids,
                                       1.5,
                                       cell=cell)
        TERMINAL_CLUSTERS = clustering.find_lj_clusters(TERMINAL_PAIRS)

        BODIES = universe.select_atoms("type 4")
        BODY_PAIRS = clustering.find_lj_pairs(BODIES.positions,
                                   BODIES.ids,
                                   1.0,
                                   cell=cell)
        body_molec_clusters = clustering.cluster_molecule_bodies(MOLECS, MOLEC_TYPES, [1, 4])
        for i, cluster in body_molec_clusters.items():
            BODY_PAIRS[i] = BODY_PAIRS[i].union(cluster)
        BODY_CLUSTERS = clustering.find_lj_clusters(BODY_PAIRS)

        # Sort the list of clusters into a consistent list so
        # we can index them.
        ALL_CLUSTERS = sorted(list(TERMINAL_CLUSTERS.union(BODY_CLUSTERS)))
        CLUSTER_POSITIONS = clustering.find_cluster_centres(ALL_CLUSTERS,
                                                 ALL_ATOMS.positions,
                                                 cutoff=10.0)
        MOLEC_TERMINALS = clustering.find_molecule_terminals(MOLECS,
                                                             atom_types=MOLEC_TYPES,
                                                             type_connections={2:[1, 4],      
                                                                               3:[1, 4],
                                                                               4:[2,3],
                                                                               1:[2, 3]})
        G = nx.Graph()
        G = clustering.connect_clusters(G, MOLEC_TERMINALS, ALL_CLUSTERS)
        ring_finder = rings.periodic_ring_finder.PeriodicRingFinder(G, CLUSTER_POSITIONS, np.array([x_size, y_size]))
        FIG, AX = plt.subplots()
        ring_finder.draw_onto(AX)
        FIG.savefig("./test_clustering_error.pdf")
        counts = Counter([len(ring) for ring in ring_finder.current_rings])
        true_counts = {8: 19, 16: 6, 4: 5, 12: 5, 20: 3, 48: 1, 44: 1}
        assert counts == true_counts




