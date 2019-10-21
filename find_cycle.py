#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:10:56 2019

@author: matthew-bailey
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def non_weighted_djikstra(node_src: int,
                          max_path_length: int,
                          node_link_id):
    """
    Calculates the shortest path to all interested
    nodes from a given source node.
    :param node_src: id of the node we wish to examine
    :param max_path_length: maximum length of a path to look for
    :param node_link_id: a list of lists of linkages, e.g.
    a square is [[1,2], [0, 3], [1, 2], [0, 3]]
    :return level_distance: the shortest distance to each node.
    -1 is a sentinel value to indicate that there is no
    path to this node shorter than max_path_length
    """
    
    level_distance = np.empty([len(node_link_id)], dtype=int)
    level_distance[:] = max_path_length + 5
    level_distance[node_src] = 0
    
    queue = [node_src]
    queue_index = 0
    queue_end = len(queue)
    while queue_index < queue_end:
        node_current = queue[queue_index]
        level_probe = level_distance[node_current] + 1
        for node_probe in node_link_id[node_current]:
            if level_distance[node_probe] > level_probe:
                level_distance[node_probe] = level_probe
                if level_probe < max_path_length:
                    queue.append(node_probe)
                    queue_end = len(queue)
        queue_index += 1
        
    level_distance[level_distance == max_path_length + 5] = -1
    return level_distance


def record_shortest_path(node_current: int,
                         level_distance,
                         node_link_id,
                         shortest_paths=None,
                         this_path=None):
    """
    Calculates the shortest path a given node according to the
    list of distances. Recursively calls iself, and returns
    a list of tuples which contain the shortest paths (there
    may be more than one)
    :param node_current: the node we want to find the shortest path to
    :param level_distance: a map of shortest distances ot nodes
    :param node_link_id: a list of lists of linkages, e.g.
    a square is [[1,2], [0, 3], [1, 2], [0, 3]]
    :param shortest_paths: used in recursion, and is returned
    at the end. Do not specify when calling yourself.
    :param this_path: used in recursion. 
    Do not specify when calling yourself.
    :return level_distance: the shortest distance to each node.
    -1 is a sentinel value to indicate that there is no
    path to this node shorter than max_path_length
    """
    if this_path is None:
        this_path = []
    if shortest_paths is None:
        shortest_paths = []
        
    level_current = level_distance[node_current]
    for node_probe in node_link_id[node_current]:
        level_probe = level_distance[node_probe]
        if level_probe == 0:
            # We've found the shortest path back to
            # the source node, so record this level.
            shortest_paths.append(tuple(this_path))
        elif level_probe == level_current - 1:
            # This node is one closer to the source, so
            # let's add it to the path.
            if len(this_path) <= (level_probe - 1):
                # The path storage list is too short, so
                # let's extend it.
                size_diff = level_probe - len(this_path)
                this_path.extend([None 
                                  for _ in range(size_diff)])
           
            this_path[level_probe - 1] = node_probe   
            record_shortest_path(node_current=node_probe,
                                 level_distance=level_distance,
                                 node_link_id=node_link_id,
                                 shortest_paths=shortest_paths,
                                 this_path=this_path)
    this_path = None
    return shortest_paths

def networkx_shortest(graph,
                      source,
                      target):
    return [_[1:-1] for _ in nx.all_shortest_paths(G,
                                             source=source,
                                             target=target)]

def find_prime_mid_nodes(level_distance,
                         ring_size: int,
                         node_link_id):
    """
    Locates all the 'prime mid nodes', i.e.
    nodes that are halfway round a ring connected to
    the current node of interest. Can find pairs of prime
    mid nodes for odd sized rings.
    :param level_distance: a list of shortest distances
    from the node (indicated by a distance of 0) to any
    other node.
    :param ring_size: the size of rings we are looking
    for midway points of
    :param node_link_id:a list of lists of linkages, e.g.
    a square is [[1,2], [0, 3], [1, 2], [0, 3]]
    :return prime_mid_nodes: a list of nodes (or pairs thereof)
    that are halfway round the rings connected to this node.
    Returns a blank list if none exist.
    """
    prime_mid_nodes = []
    midway_point = ring_size // 2
    midway_nodes = np.argwhere(level_distance == midway_point).ravel()
    for node in midway_nodes:
        linked_nodes = node_link_id[node]
        # Check for pairs of prime mid nodes, where this node
        # is connected to another node the same distance from
        # the source.
        for linked_node in linked_nodes:
            if level_distance[linked_node] == midway_point:
                prime_mid_nodes.append([node, linked_node])
                
        # Check for a single prime mid node. There is obvious
        # if this node is connected to two others which are
        # one unit closer to the source
        connected_shorter = level_distance[linked_nodes] == midway_point - 1
        if np.sum(connected_shorter) == 2:
            prime_mid_nodes.append([node])
    return prime_mid_nodes


def construct_ring_graoh(rings):
G = nx.Graph()
G.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],  [0, 5],
                  [5, 6], [6, 7], [7, 8], [8, 9], [9, 0],
                  [9, 10], [10, 11], [11, 12], [12, 13], [13, 1]])
    
def get_num_links(graph):
    return [len([_ for _ in nx.all_neighbors(graph, node)]) 
            for node in graph.nodes()]
def get_neighbours(graph):
    return [[_ for _ in nx.all_neighbors(graph, node)]
            for node in graph.nodes()]
print([i for i in range(12)])
distances = non_weighted_djikstra(0, 3, get_neighbours(G))
mid_prime_nodes = find_prime_mid_nodes(distances, 6, get_neighbours(G))
for mid_prime_list in mid_prime_nodes:
    for mpn in mid_prime_list:
        print(mpn)
        shortest_path = record_shortest_path(node_current=mpn,
                             level_distance=distances,
                             node_link_id=get_neighbours(G))
        print(networkx_shortest(G, 0, mpn))
                             
                             

nx.draw(G)