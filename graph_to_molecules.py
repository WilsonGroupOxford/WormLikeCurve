#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:01:29 2020

@author: matthew-bailey
"""

import networkx as nx
from WormLikeCurve.WormLikeCurve import WormLikeCurve
from WormLikeCurve.CurveCollection import CurveCollection
from WormLikeCurve.Bonds import HarmonicBond, AngleBond
import matplotlib.pyplot as plt
import numpy as np

def graph_to_molecules(graph: nx.Graph, pos, edge_factor = 0.1,
                       num_segments = 5, periodic_box=None):
    """
    Converts a graph to a set of molecules.
    TODO: fix periodicity by changing the starting
    position. hope LAMMPS sorts the rest 
    of it out!
    """
    molecules = []
    for u, v in graph.edges():
        gradient = pos[v] - pos[u]
        has_changed_x, has_changed_y = False, False
        if periodic_box is not None:
            minimum_image_x = (periodic_box[0, 1] - periodic_box[0, 0]) / 2
            minimum_image_y = (periodic_box[1, 1] - periodic_box[1, 0]) / 2
            if gradient[0] > minimum_image_x:
                gradient[0] -= minimum_image_x * 2
                has_changed_x = True
            elif gradient[0] < -minimum_image_x:
                gradient[0] += minimum_image_x * 2
                has_changed_x = True
            
            if gradient[1] > minimum_image_y:
                gradient[1] -= minimum_image_y * 2
                has_changed_y = True
            elif gradient[1] < -minimum_image_y:
                gradient[1] += minimum_image_y * 2
                has_changed_y = True
        normalised_gradient = gradient / np.sqrt(np.dot(gradient, gradient))
        angle = np.arccos(np.dot(normalised_gradient, np.array([1.0, 0.0])))
        if has_changed_x:
            print(gradient, " has changed in x. Was ", pos[v] - pos[u])
            angle += 0.0
        if has_changed_y:
            print(gradient, " has changed in y. Was ", pos[v] - pos[u])
            angle = -angle
        starting_point = pos[u] + (edge_factor * gradient)
        segment_length = (1 - 2 * edge_factor) * gradient / num_segments
        segment_length = np.hypot(segment_length[0], segment_length[1])
        harmonic_bond = HarmonicBond(k=1, length=segment_length)
        angle_bond = AngleBond(k=100, angle=np.pi)
        curve = WormLikeCurve(num_segments=num_segments,
                              harmonic_bond=harmonic_bond,
                              angle_bond=angle_bond,
                              start_pos=starting_point)
        curve.start_pos = starting_point
        curve.vectors = np.array([[segment_length, angle] for i in range(num_segments)])
        curve.positions = curve.vectors_to_positions()
        molecules.append(curve)
    return CurveCollection(molecules)


if __name__ == "__main__":
    num_nodes: int = 6
    hex_graph = nx.generators.lattice.hexagonal_lattice_graph(num_nodes, num_nodes, periodic=True)
    pos = dict(nx.get_node_attributes(hex_graph, 'pos'))
    for key, val in pos.items():
        val = np.array(val)
        pos[key] = val
    curves = graph_to_molecules(hex_graph, pos,periodic_box=np.array([[0.0, 8.866],[0.0, 8.65]]))
    fig, ax = plt.subplots()
    nx.draw(hex_graph, pos=pos, with_labels=True, ax=ax)
    curves.plot_onto(ax)
