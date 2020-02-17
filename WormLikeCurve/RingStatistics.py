#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:17:37 2019

@author: matthew-bailey
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


class RingStatistics:
    """
    Class to analyse the rings from a networkx
    graph.
    """

    def __init__(self, graph: nx.Graph, max_ring_size: int = 10):
        """
        Initialise the RingStatistics finder.
        :param graph: a networkx graph object
        :param max_ring_size: the maximum size of ring
        we wish to look for. The running time of
        this code scales O(N^3) with this number.
        """
        self.graph = graph
        self.max_ring_size = max_ring_size
        self.path_cutoff = max_ring_size // 2

        self.shortest_path_dict = None

    def all_shortest_paths(self, recalculate: bool = False):
        """
        Calculates the shortest path between all pairs, up to
        the path cutoff specified in the initialiser.
        Memoises the list of shortest paths to save time
        in the other routines.
        :param recalculate: if the graph object has
        changed, force this to recalculate everything.
        :return lengths : Dictionary, keyed by source and target,
        of shortest paths.
        """
        if self.shortest_path_dict is None or recalculate:
            shortest_paths = nx.all_pairs_shortest_path(self.graph, self.path_cutoff)
            self.shortest_path_dict = dict(shortest_paths)
        return self.shortest_path_dict

    def find_prime_mid_nodes(self, source_node: int, ring_size: int):
        """
        Finds all the 'prime mid nodes' around
        a given source node for rings of a specific size.
        A prime mid node is a node that is exactly half
        way round a ring from our source node.
        :param source_node: the node around which we want
        to find the rings
        :param ring_size: the size of the rings to look for.
        :return prime_mid_node: a list of nodes
        potentially halfway round a ring from the source node
        or a list of pairs of nodes if the ring size is odd
        """
        # Remember this dict contains paths of nodes, not
        # lengths. Subtract one to get the length.
        lengths = {
            key: len(value) - 1
            for key, value in self.shortest_path_dict[source_node].items()
        }
        # Check which nodes are potentially halfway round a ring.
        interesting_nodes = []
        for node, length in lengths.items():
            if length == ring_size // 2:
                interesting_nodes.append(node)

        # If we're part of an even size ring,
        # there is one prime mid node with two
        # neighbours of path length (N/2) - 1
        prime_mid_nodes = []
        if (ring_size % 2) == 0:
            for node in interesting_nodes:
                num_shorter_neighbors = 0
                for neighbor in self.graph.neighbors(node):
                    if lengths[neighbor] == (ring_size // 2) - 1:
                        num_shorter_neighbors += 1

                if num_shorter_neighbors == 2:
                    prime_mid_nodes.append(node)
        # If we're part of an odd size ring, there is a pair
        # of prime mid nodes, each with one neighbour the
        # same distance from the source node.
        else:
            for node in interesting_nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in interesting_nodes:
                        prime_mid_nodes.append([node, neighbor])
                        interesting_nodes.remove(neighbor)
        return prime_mid_nodes

    def find_ring(self, source_node: int, mid_nodes, ring_size):
        """
        Constructs the that includes a given source node
        and the prime mid node on the other side.
        :param source_node: the node that we want to analyse
        :param mid_nodes: the node (or pair thereof) of nodes
        halfway round the ring.
        """
        try:
            iter(mid_nodes)
        except TypeError:
            # It's just an integer.
            # Box it into an iterable.
            mid_nodes = [mid_nodes]

        # This is an odd size ring, so there is only one
        # shortest path.
        if len(mid_nodes) == 2:
            shortest_paths = [None, None]
            shortest_paths[0] = self.shortest_path_dict[source_node][mid_nodes[0]]
            shortest_paths[1] = self.shortest_path_dict[source_node][mid_nodes[1]]
        else:
            shortest_paths = nx.all_shortest_paths(
                self.graph, source_node, mid_nodes[0]
            )
        # Now flatten this list, we'll rearrange it into a ring
        # shortly.
        proto_ring = [item for sublist in shortest_paths for item in sublist]
        return self.reassemble_ring(proto_ring, ring_size)

    def reassemble_ring(self, proto_ring, ring_size):
        """
        Turns a 'proto ring', i.e. a list of nodes that are
        in a ring, into a ring that we can follow which is
        ordered consistently.
        :param proto_ring: a list of nodes, can be duplicates, that are
        in our ring in any order.
        :param ring_size: the size of the ring we want to look for,
        used to check sanity at the end.
        :return ring: a ring of connected bonds, starting from
        the smallest index, which then connects to its next smallest
        indexed neighbor. This is an empty tuple if the proto_ring
        cannot be reassembled sanely.
        """
        # We need the rings to be ordered consistently,
        # so we can easily test them for uniqueness.
        # Start off at the smallest indexed node of this ring.
        start_node = min(proto_ring)
        this_node = start_node
        ring = [this_node]
        while True:
            # Check the neighbors of this ring in size order,
            # and move to the smallest we find that is in
            # the 'proto ring'. Also make sure that we are
            # not stepping backwards to a node we have already
            # visited.
            for neighbor in sorted(self.graph.neighbors(this_node)):
                if neighbor in proto_ring and neighbor not in ring:
                    this_node = neighbor
                    ring.append(this_node)
                    break
            else:
                # We've looked at all the neighbors but none have met
                # our criteria -- we've either returned to the start
                # or snarled something up. Either way, time to bail out.
                break

        # Uh oh -- we know the size of ring we're looking for, and this
        # isn't it. This is most commonly a problem when the program
        # has picked up a tennis racket shaped ring -O and wants to
        # walk the same edge twice. Throw this out by returning
        # an empty ring.
        if len(ring) != ring_size:
            return tuple()
        return tuple(ring)

    def is_primitive_ring(self, ring):
        """
        Calculates if a ring is primitive (i.e. cannot be broken
        up into smaller rings). Checks all opposite pairs
        in the ring to see if the ring path is the
        shortest path.
        :param ring: an ordered list of nodes that make up the ring.
        :return is_primitive: true if this is a primitive ring,
        false if a shorter path can be found.
        """

        if not ring:
            return False

        ring_size = len(ring)
        midway = ring_size // 2
        for i in range(midway):
            shortest_path = self.shortest_path_dict[ring[i]][ring[i + midway]]
            if len(shortest_path) - 1 < midway:
                return False
        return True

    def find_rings_of_size(self, ring_size: int):
        """
        Finds all of the rings of size ring_size.
        :param ring_size: an integer specifying the size
        of ring you wish to find.
        :return rings: a set of tuples, with the tuple
        being an ordered list of ring vertices starting
        at the smallest numbered vertex and moving on
        to the next smallest numbered vertex.
        """
        rings = set()
        for node in self.graph.nodes():
            prime_mid_nodes = self.find_prime_mid_nodes(node, ring_size)
            for mid_node in prime_mid_nodes:
                ring = self.find_ring(node, mid_node, ring_size)
                if ring not in rings:
                    if self.is_primitive_ring(ring):
                        rings.add(ring)
        return rings

    def find_all_rings(self):
        """
        Finds all of the rings of all possible sizes
        between 3 and the maximum ring size.
        Calls find_rings_of_size function repeatedly.
        :return rings: a set of tuples, with the tuple
        being an ordered list of ring vertices starting
        at the smallest numbered vertex and moving on
        to the next smallest numbered vertex.
        """
        rings = set()
        for ring_size in range(3, self.max_ring_size):
            rings_n = self.find_rings_of_size(ring_size)
            for item in rings_n:
                rings.add(item)
        return rings


def assemble_ring_graph(rings):
    """
    Calculates which rings share edges, and represents this
    as a networkx graph. Scales O(N^2) with the number of rings,
    and also with the size of the largest ring. Ring indicies
    correspond to the order in a shortest list according to
    python's default sorting.
    :param rings: a set of tuples, with the tuples being
    ring orders according to reassemble_ring.
    :return graph: a networkx graph with a connection
    representing two rings sharing an edge.
    """
    graph = nx.Graph()
    ring_list = sorted(list(rings))
    for i, ring in enumerate(ring_list):
        # Each ring is made up of pairs of edges.
        # First, find those pairs of edges using modulo
        # arithmetic.
        edge_pairs = []
        for edge_idx in range(len(ring)):
            other_edge_idx = (edge_idx + 1) % len(ring)
            edge = ring[edge_idx]
            other_edge = ring[other_edge_idx]
            edge_pairs.append([edge, other_edge])
        for j, other_ring in enumerate(ring_list[i + 1 :], i + 1):
            # We know that edge and other_edge are connected
            # because they're in a ring. The shortest path in
            # any given ring must be via this edge. If both
            # edges are in another ring, then this ring
            # shares an edge with another ring.
            for edge, other_edge in edge_pairs:
                if edge in other_ring and other_edge in other_ring:
                    graph.add_edge(i, j)
                    break
    return graph


def identify_degenerate_rings(graph, rings):
    """
    Checks for degenerate rings, which are
    those that are composed of smaller rings.

    """
    node_coordinations = [len(list(graph.neighbors(node))) for node in graph.nodes]

    ring_visit_count = Counter()
    for ring in rings:
        ring_visit_count.update(ring)

    good_rings = set()
    for ring in rings:
        for node in ring:
            if ring_visit_count[node] == node_coordinations[node]:
                good_rings.add(ring)
                break
    return good_rings


if __name__ == "__main__":
    G = nx.Graph()
    with open("./node_cnxs.dat", "r") as FI:
        for LINE in FI.readlines():
            LINE = LINE.split()
            G.add_edge(int(LINE[0]), int(LINE[1]))
    COORDINATES = dict()
    with open("./node_crds.dat", "r") as FI:
        for I, LINE in enumerate(FI.readlines()):
            COORDINATES[I] = [float(ITEM) for ITEM in LINE.split()]
    FIG, AX = plt.subplots()
    nx.draw(G, with_labels=True, pos=COORDINATES, ax=AX)
    RS = RingStatistics(G, 50)
    RS.all_shortest_paths()
    RINGS = RS.find_all_rings()
    DAVID_RINGS = set()
    plt.close(FIG)
    with open("./rings.dat", "r") as fi:
        for LINE in fi.readlines():
            THIS_RING = [int(I) for I in LINE.split()]
            THIS_RING = RS.reassemble_ring(THIS_RING, len(THIS_RING))
            DAVID_RINGS.add(THIS_RING)
    ring_graph = assemble_ring_graph(DAVID_RINGS)
    DAVID_RINGS_SORTED = sorted(list(DAVID_RINGS))
    RING_CENTROIDS = {}

    for I, RING in enumerate(DAVID_RINGS_SORTED):
        POSITIONS = np.empty([len(RING), 2])
        for J, NODE in enumerate(RING):
            POSITIONS[J, :] = COORDINATES[NODE]
        RING_CENTROIDS[I] = np.mean(POSITIONS, axis=0)

    FIG, AX = plt.subplots()
    nx.draw(ring_graph, with_labels=True, pos=RING_CENTROIDS)
    GOOD_RINGS = identify_degenerate_rings(G, RINGS)
    print(len(GOOD_RINGS), len(DAVID_RINGS))
