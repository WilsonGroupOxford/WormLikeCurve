#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:34:20 2020

@author: newc4592
"""

from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from rings import PeriodicRingFinder

import sys

UNKNOWN_COLOUR = (2, 3)
COLOUR_TO_TYPE = defaultdict(lambda: UNKNOWN_COLOUR)
COLOUR_TO_TYPE[0] = (2,)
COLOUR_TO_TYPE[1] = (3,)
CORRESPONDING_COLOURS = {2: 3, 3: 2, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
COLOUR_LUT = {
    None: "blue",
    0: "white",
    1: "purple",
    2: "blue",
    3: "green",
    4: "orange",
    5: "red",
    6: "purple",
    7: "pink",
    8: "brown",
    9: "cyan",
    10: "magenta",
    11: "yellow",
}


def draw_nonperiodic_coloured(graph: nx.Graph, pos: Dict[int, np.array], ax=None):
    """
    Draw an aperiodic graph with the nodes coloured correctly.

    :param graph: the graph we wish to draw with node attributes of 'color'
    :param pos: a dictionary keyed by node id with values being positions
    :param periodic_box: the periodic box to wrap raound
    :param ax: the axis to draw on. Can be none for a fresh axis.
    :return: an axis with the drawn graph on.
    """
    if ax is None:
        _, ax = plt.subplots()
    edge_list = []
    for u, v in graph.edges():
        edge_list.append((u, v))
    nodes_in_edge_list = set([item for edge_pair in edge_list for item in edge_pair])
    nodes_in_edge_list = list(nodes_in_edge_list)

    node_colours = {
        node_id: (COLOUR_LUT[colour[0]] if colour is not None else "blue")
        for node_id, colour in graph.nodes(data="color")
    }
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        node_size=10,
        node_color=[node_colours[node_id] for node_id in nodes_in_edge_list],
        edgelist=edge_list,
        nodelist=nodes_in_edge_list,
        font_size=8,
        edgecolors="black",
        linewidths=0.5,
    )

    return ax


def draw_periodic_coloured(
    graph: nx.Graph, pos: Dict[int, np.array], periodic_box: np.array, ax=None, **kwargs
):
    """
    Draw a periodic graph with the nodes coloured correctly.

    :param graph: the graph we wish to draw with node attributes of 'color'
    :param pos: a dictionary keyed by node id with values being positions
    :param periodic_box: the periodic box to wrap raound
    :param ax: the axis to draw on. Can be none for a fresh axis.
    :return: an axis with the drawn graph on.
    """
    if ax is None:
        _, ax = plt.subplots()
    edge_list = []
    periodic_edge_list = []
    for u, v in graph.edges():
        distance = np.abs(pos[v] - pos[u])
        if (
            distance[0] < periodic_box[0, 1] / 2
            and distance[1] < periodic_box[1, 1] / 2
        ):
            edge_list.append((u, v))
        else:
            periodic_edge_list.append((u, v))
    nodes_in_edge_list = set([item for edge_pair in edge_list for item in edge_pair])
    nodes_in_edge_list = list(nodes_in_edge_list)

    periodic_nodes = set(
        [item for edge_pair in periodic_edge_list for item in edge_pair]
    )
    periodic_nodes = list(periodic_nodes)

    try:
        node_colours = {
            node_id: (COLOUR_LUT[colour[0]] if colour is not None else "blue")
            for node_id, colour in graph.nodes(data="color")
        }
    except TypeError:
        node_colours = {
            node_id: (COLOUR_LUT[colour] if colour is not None else "blue")
            for node_id, colour in graph.nodes(data="color")
        }
    nx.draw(
        graph,
        pos=pos,
        ax=ax,
        node_size=10,
        node_color=[node_colours[node_id] for node_id in nodes_in_edge_list],
        edgelist=edge_list,
        nodelist=nodes_in_edge_list,
        # font_size=8,
        edgecolors="black",
        linewidths=0.5,
        **kwargs,
    )

    node_colours_list = []
    new_edge_list = []
    new_node_list = []
    new_pos = {key: value for key, value in pos.items()}
    temporary_edges = []
    temporary_nodes = []
    # We often encounter an edge periodic in more than one
    # way. Keep track, and give each one a virtual position.
    encounters = defaultdict(lambda: 0)
    for u, v in periodic_edge_list:
        gradient = new_pos[v] - new_pos[u]

        # If we're in a periodic box, we have to apply the
        # minimum image convention. Do this by creating
        # a virtual position for v, which is a box length away.
        minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
        minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
        # print(pos[v], pos[u])
        # print(gradient, minimum_image_x, minimum_image_y)
        # We need the += and -= to cope with cases where we're out in
        # both x and y.
        new_pos_v = np.array([item for item in new_pos[v]])
        if gradient[0] > minimum_image_x:
            new_pos_v -= np.array([2 * minimum_image_x, 0.0])
        elif gradient[0] < -minimum_image_x:
            new_pos_v += np.array([2 * minimum_image_x, 0.0])

        if gradient[1] > minimum_image_y:
            new_pos_v -= np.array([0, 2 * minimum_image_y])
        elif gradient[1] < -minimum_image_y:
            new_pos_v += np.array([0, 2 * minimum_image_y])

        encounters[v] += 1
        new_v = f"{v}_periodic_{encounters[v]}"
        new_pos[new_v] = new_pos_v
        node_colours[new_v] = node_colours[v]
        node_colours_list.extend([node_colours[node_id] for node_id in (u, new_v)])
        new_edge_list.append((u, new_v))
        new_node_list.extend([u, new_v])
        temporary_edges.append((u, new_v))
        temporary_nodes.append(new_v)

    graph.add_edges_from(temporary_edges)

    nx.draw(
        graph,
        pos=new_pos,
        ax=ax,
        node_size=10,
        node_color=node_colours_list,
        edgelist=new_edge_list,
        nodelist=new_node_list,
        style="dashed",
        edgecolors="black",
        linewidths=0.5,
    )

    graph.remove_edges_from(temporary_edges)
    graph.remove_nodes_from(temporary_nodes)
    return ax


def colour_graph(graph: nx.Graph, colour_to_type: Dict = COLOUR_TO_TYPE) -> nx.Graph:
    """
    Assign a type to each node of a graph.

    Proceeds recursively, assigning a type to each node on a graph.
    Then, assigns the corresponding type according to the type dictionary.
    In the case of odd rings, this can't be done, so we instead assign
    a set of types to that node.

    :param graph: the graph to colour
    :param corresponding_types: a dictionary with a set of types in it,
    each of which corresponds to one other.
    """
    colours = nx.algorithms.coloring.greedy_color(
        graph, strategy="smallest_last", interchange=True
    )
    for key, value in colours.items():
        colours[key] = colour_to_type.get(value, UNKNOWN_COLOUR)
    nx.set_node_attributes(graph, colours, "color")
    return graph


def load_morley(
    prefix: str, reset_origin: bool = False
) -> Tuple[Dict[int, np.array], nx.Graph, np.array]:

    coords_file = prefix + "_crds.dat"
    network_file = prefix + "_net.dat"
    aux_file = prefix + "_aux.dat"

    graph = nx.Graph()
    pos_dict = dict()
    with open(coords_file) as fi:
        for i, line in enumerate(fi.readlines()):
            coords = [float(item) for item in line.split()]
            pos_dict[i] = np.array(coords)

    with open(network_file) as fi:
        for u, line in enumerate(fi.readlines()):
            connections = [int(item) for item in line.split()]
            for v in connections:
                graph.add_edge(u, v)

    dual_file = prefix + "_dual.dat"
    dual_connections = dict()
    with open(dual_file) as fi:
        for node_id, line in enumerate(fi.readlines()):
            dual_connections[node_id] = [int(item) for item in line.split()]

    nx.set_node_attributes(graph, dual_connections, "dual_connections")

    with open(aux_file) as fi:
        num_atoms = int(fi.readline())
        _, _ = [int(item) for item in fi.readline().split()]
        geometry_code = fi.readline().strip()
        box_max_x, box_max_y = [float(item) for item in fi.readline().split()]
        inv_box_max_x, inv_box_max_y = [float(item) for item in fi.readline().split()]

        if not np.isclose(box_max_x, 1.0 / inv_box_max_x):
            raise RuntimeError(
                "Inverse periodic box side does not match periodic box size."
            )

        if not np.isclose(box_max_y, 1.0 / inv_box_max_y):
            raise RuntimeError(
                "Inverse periodic box side does not match periodic box size."
            )
        periodic_box = np.array([[0.0, box_max_x], [0.0, box_max_y]], dtype=float)

    if reset_origin:
        min_x = min(val[0] for val in pos_dict.values())
        min_y = min(val[1] for val in pos_dict.values())
        for key in pos_dict.keys():
            pos_dict[key] += np.array([min_x, min_y])
    graph = colour_graph(graph)
    return pos_dict, graph, periodic_box


def construct_morley_dual(
    graph: nx.Graph, pos: Dict[int, np.array], periodic_box: np.array
):
    ring_finder = PeriodicRingFinder(graph, pos, periodic_box[:, 1])

    num_nodes = len(graph)
    dual_connections = defaultdict(list)
    current_rings = list(ring_finder.current_rings)
    for ring_id, ring in enumerate(current_rings):
        for node in ring.to_node_list():
            real_node = node % num_nodes
            dual_connections[real_node].append(ring_id)

    # Now we must order the dual connections clockwise from +y
    sorted_dual_connections = dict()
    for node, node_duals in dual_connections.items():
        node_pos = pos[node]
        duals_pos = [current_rings[ring_id].centroid() for ring_id in node_duals]
        duals_vectors = [pos - node_pos for pos in duals_pos]
        # Apply the minimum image convention
        for vec in duals_vectors:
            minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
            minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
            if vec[0] > minimum_image_x:
                vec -= np.array([2 * minimum_image_x, 0.0])
            elif vec[0] < -minimum_image_x:
                vec += np.array([2 * minimum_image_x, 0.0])

            if vec[1] > minimum_image_y:
                vec -= np.array([0, 2 * minimum_image_y])
            elif vec[1] < -minimum_image_y:
                vec += np.array([0, 2 * minimum_image_y])
        duals_vectors = [vec / np.linalg.norm(vec) for vec in duals_vectors]
        angles_with_y = [np.sign(vec[0]) * np.arccos(vec[1]) for vec in duals_vectors]
        angles_with_y = [
            2 * np.pi + angle if angle < 0 else angle for angle in angles_with_y
        ]
        sorted_indices = np.argsort(angles_with_y)
        sorted_node_duals = [node_duals[i] for i in sorted_indices]
        sorted_dual_connections[node] = sorted_node_duals

    dual_connections = sorted_dual_connections
    morley_connections = nx.get_node_attributes(graph, "dual_connections")
    return dual_connections


def write_out_morley(graph: nx.Graph, pos, periodic_box: np.array, prefix: str):
    """
    Write out into a netmc readable file
    """
    coordinate_file = prefix + "_crds.dat"
    aux_file = prefix + "_aux.dat"
    network_file = prefix + "_net.dat"
    dual_file = prefix + "_dual.dat"

    with open(coordinate_file, "w") as fi:
        for i in range(len(graph)):
            fi.write(f"{pos[i][0]}\t{pos[i][1]}\n")

    all_dual_connections = nx.get_node_attributes(graph, "dual_connections")
    with open(aux_file, "w") as fi:
        fi.write("f{len(graph)}\n")  # Number of nodes
        max_net_connections = max(graph.degree())[1]
        dual_connections = [len(val) for key, val in all_dual_connections.items()]
        max_dual_connections = max(dual_connections)
        fi.write(f"{max_net_connections}\t{max_dual_connections}\n")
        fi.write("2DE\n")  # Geometry code
        x_size = np.abs(periodic_box[0, 1] - periodic_box[0, 0])
        y_size = np.abs(periodic_box[1, 1] - periodic_box[1, 0])
        fi.write(f"{x_size} \t {y_size} \n")  # Periodic box
        fi.write(f"{1.0/x_size} \t {1.0 / y_size} \n")  # Reciprocal box

    with open(network_file, "w") as fi:
        for i in range(len(graph)):
            neighbours = graph.neighbors(i)
            fi.write("\t".join([str(item) for item in neighbours]) + "\n")

    with open(dual_file, "w") as fi:
        for i in range(len(graph)):
            dual_connections = all_dual_connections[i]
            fi.write("\t".join([str(item) for item in dual_connections]) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        MORLEY_PREFIX = sys.argv[1]
    else:
        MORLEY_PREFIX = "./Data/hexagon_network_A"
    HEX_POS, HEX_GRAPH, HEX_BOX = load_morley(MORLEY_PREFIX)
    print("HEX_POS has", len(HEX_POS), "items at the start.")
    DUAL_CNXS = construct_morley_dual(HEX_GRAPH, pos=HEX_POS, periodic_box=HEX_BOX)
    nx.set_node_attributes(HEX_GRAPH, DUAL_CNXS, name="dual_connections")
    write_out_morley(HEX_GRAPH, HEX_POS, HEX_BOX, prefix="./Data/hexagon_rewritten_A")
