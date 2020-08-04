from rings.ring_finder import RingFinder
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def load_positions(filename):
    pos_dict = {}
    with open(filename) as fi:
        for i, line in enumerate(fi.readlines()):
            pos_dict[i] = np.array([float(item.strip()) for item in line.split()])
    return pos_dict


def load_edge_list(filename):
    edge_list = []
    with open(filename) as fi:
        for line in fi:
            line = [int(item.strip()) for item in line.split()]
            for i in range(1, len(line)):
                edge_list.append((line[0], line[i]))
    return edge_list


def load_graph(prefix):
    pos_dict = load_positions(f"{prefix}_coordinates.out")
    edge_list = load_edge_list(f"{prefix}_connectivity.out")
    network = nx.Graph()
    network.add_edges_from(edge_list)
    nx.set_node_attributes(network, pos_dict, name="pos")
    return network


if __name__ == "__main__":
    if len(sys.argv) == 2:
        PREFIX = sys.argv[1]
    GRAPH = load_graph(PREFIX)
    rf = RingFinder(GRAPH, coords_dict=nx.get_node_attributes(GRAPH, "pos"))
    FIG, AX = plt.subplots()
    rf.draw_onto(
        ax=AX, cmap_name="coolwarm", color_by="regularity", color_reversed=True
    )
    AX.set_axis_off()
    FIG.savefig(f"{PREFIX}.pdf", bbox_inches="tight")
    with open(f"{PREFIX}_regularities.csv", "w") as fi:
        fi.write("Sides, Convexity, Solidity, BalancedRepartition, Regularity\n")
        for shape in rf.current_rings:
            fi.write(
                f"{len(shape)}, {shape.convexity_metric()}, {shape.solidity_metric()}, {shape.balanced_repartition_metric()}, {shape.regularity_metric()}\n"
            )
    print(np.mean([shape.regularity_metric() for shape in rf.current_rings]))
