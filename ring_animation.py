#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:50:08 2021

@author: matthew-bailey
"""
from graph_to_molecules import hexagonal_lattice_graph
from rings.periodic_ring_finder import PeriodicRingFinder
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    HEX_GRAPH = hexagonal_lattice_graph(6, 6)
    periodic_box = np.array([[0.0, 1.5 * 6], [0.0, 6 * np.sqrt(3)]])
    HEX_GRAPH = nx.convert_node_labels_to_integers(HEX_GRAPH, label_attribute="label")
    POS = nx.get_node_attributes(HEX_GRAPH, "pos")
    for node in HEX_GRAPH:
        print(node)
    rf = PeriodicRingFinder(HEX_GRAPH,
                            coords_dict=POS,
                            cell=periodic_box)