#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:09:24 2020

@author: matthew-bailey
"""

import sys
import os
import copy

import heapq

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats


from collections import Counter

from morley_parser import load_morley, draw_periodic_coloured
from rings.periodic_ring_finder import PeriodicRingFinder
from rings.ring_finder import convert_to_ring_graph, topological_rdf,RingFinderError, geometric_rdf
from rings.shape import Shape
import nodeme


def normalise_counter(in_count):
    """
    Normalise a counter so the sum of all its elements is 1.
    """
    total = sum(in_count.values(), 0.0)
    for key in in_count:
        in_count[key] /= total
    return in_count


def find_powerlaw_fit(xs, ys):
    """
    Find the coefficients that match y = Ax^-b
    """
    ln_xs = np.log(xs)
    ln_ys = np.log(ys)
    # for the fit to work we have to discard all the invalid entries
    valid_entries = np.logical_and(np.logical_and(~np.isnan(ln_xs), ~np.isinf(ln_xs)), np.logical_and(~np.isnan(ln_ys), ~np.isinf(ln_ys)))

    ln_xs = ln_xs[valid_entries]
    ln_ys = ln_ys[valid_entries]
    A = np.vstack([ln_xs, np.ones(len(ln_xs))]).T
    grad, intercept = np.linalg.lstsq(A, ln_ys, rcond=None)[0]
    return grad, intercept


def li_smax_graph(degree_seq, create_using=None):
    """Generates a graph based with a given degree sequence and maximizing
    the s-metric.  Experimental implementation.

    Maximum s-metrix  means that high degree nodes are connected to high
    degree nodes. 
        
    - `degree_seq`: degree sequence, a list of integers with each entry
       corresponding to the degree of a node.
       A non-graphical degree sequence raises an Exception.    
    """
    degree_seq.sort() # make sure it's sorted
    degree_seq.reverse()
    degrees_left = degree_seq[:]

    A_graph = nx.empty_graph(0,create_using)
    A_graph.add_node(0)
    a_list = [False]*len(degree_seq)
    b_set = set(range(1,len(degree_seq)))
    a_open = set([0])
    O = []
    for j in b_set:
        heapq.heappush(O, (-degree_seq[0]*degree_seq[j], (0,j)))
    wa = degrees_left[0] #stubs in a_graph
    db = sum(degree_seq) - degree_seq[0] #stubs in b-graph
    a_list[0] = True #node 0 is now in a_Graph
    bsize = len(degree_seq) -1 #size of b_graph
    selected = []
    weight = 0
    while O or selected:
        if len(selected) <1 :
            firstrun = True
            while O:
                (newweight, (i,j)) = heapq.heappop(O)
                if degrees_left[i] < 1 or degrees_left[j] < 1 :
                    continue
                if firstrun:
                    firstrun = False
                    weight = newweight
                if not newweight == weight:
                    break
                heapq.heappush(selected, [-degrees_left[i], \
                                    -degrees_left[j], (i,j)])
            if not weight == newweight:
                heapq.heappush(O,(newweight, (i,j)))
            weight *= -1
        if len(selected) < 1:
            break
        
        [w1, w2, (i,j)] = heapq.heappop(selected)
        if degrees_left[i] < 1 or degrees_left[j] < 1 :
            continue
        if a_list[i] and j in b_set:
            #TYPE1
            a_list[j] = True
            b_set.remove(j)
            A_graph.add_node(j)
            A_graph.add_edge(i, j)
            degrees_left[i] -= 1
            degrees_left[j] -= 1
            wa += degree_seq[j] - 2
            db -= degree_seq[j]
            bsize -= 1
            newweight = weight
            if not degrees_left[j] == 0:
                a_open.add(j)
                for k in b_set:
                    if A_graph.has_edge(j, k): continue
                    w = degree_seq[j]*degree_seq[k]
                    if w > newweight:
                        newweight = w
                    if weight == w and not newweight > weight:
                        heapq.heappush(selected, [-degrees_left[j], \
                                            -degrees_left[k], (j,k)])
                    else:
                        heapq.heappush(O, (-w, (j,k)))
                if not weight == newweight:
                    while selected:
                        [w1,w2,(i,j)] = heapq.heappop(selected)
                        if degrees_left[i]*degrees_left[j] > 0:
                            heapq.heappush(O, [-degree_seq[i]*degree_seq[j],(i,j)])
            if degrees_left[i] == 0:
                a_open.discard(i)
                    
        else:
            #TYPE2
            if db == (2*bsize - wa):
                #tree condition
                continue
            elif db < 2*bsize -wa:
                raise nx.NetworkXError(\
                        "THIS SHOULD NOT HAPPEN!-not graphable")
                continue
            elif wa == 2 and bsize > 0:
                #disconnected cluster condition
                continue
            elif wa == db - (bsize)*(bsize-1):
                continue
            A_graph.add_edge(i, j)
            degrees_left[i] -= 1
            degrees_left[j] -= 1
            if degrees_left[i] < 1:
                a_open.discard(i)
            if degrees_left[j] < 1:
                a_open.discard(j)
            wa -=  2
            if not degrees_left[i] < 0 and not degrees_left[j] < 0:
                selected2 = (selected)
                selected = []
                while selected2:
                    [w1,w1, (i,j)] = heapq.heappop(selected2)
                    if degrees_left[i]*degrees_left[j] > 0:
                        heapq.heappush(selected, [-degrees_left[i], \
                                        -degrees_left[j], (i,j)])
    return A_graph 


def generate_ring_size_fig(ring_sizes, angle, length):
    ring_size_arr = np.vstack(ring_sizes)
    mean_ring_sizes = np.mean(ring_size_arr, axis=0)
    ring_size_idx = np.array([i for i in range(ring_size_arr.shape[1])])
    grad, intercept = find_powerlaw_fit(ring_size_idx, mean_ring_sizes)
    mean_ring_size = 6
    mode_idx, mode_height = np.argmax(mean_ring_sizes), np.max(mean_ring_sizes)
    _maxent = nodeme.NodeME(k_limits=(3, 20), k_mean=mean_ring_size)
    maxent_pk = _maxent(target_pk=mode_height, k=mode_idx)

    rsfig, rsax = plt.subplots()
    # rsax.bar(x=ring_size_idx, height=mean_ring_sizes, width=1.0)

    rsax.errorbar(x=ring_size_idx, y=mean_ring_sizes, yerr=np.std(ring_size_arr, axis=0, ddof=1) / np.sqrt(ring_size_arr.shape[1]), capsize=5.0)
    rs_smooth = np.linspace(0, ring_size_arr.shape[0], 50)
    rs_powerlaw = np.exp(intercept) * np.linspace(0, ring_size_arr.shape[0], 50)**(grad)
    rs_d, rs_p = stats.ks_2samp(rs_smooth, mean_ring_sizes)
    rsax.plot(rs_smooth, rs_powerlaw, linestyle="dashed", color="black", label=f"s-f, p={rs_p:.2f}")
    if maxent_pk is not None:
        rs_maxent_d, rs_maxent_p = stats.ks_2samp(maxent_pk, mean_ring_sizes)
        rsax.plot(_maxent.k, maxent_pk, linestyle="dotted", color="red", label=f"m-e, p={rs_maxent_p:.2f}")
    
    rsax.set_xlim(0, 10)
    rsax.set_ylim(0, 0.5)
    rsax.set_xlabel("ring size r")
    rsax.set_ylabel("p(r)")
    rsax.legend()
    filename = f"./Results/Ring_Sizes/RINGSIZES_{angle}_{length}.pdf"
    print("saving fig to", filename)
    rsfig.savefig(filename)
    plt.close(rsfig) 

def generate_node_degree_fig(node_degrees, angle, length):
    print("Generating node degree fig")
    node_degrees_arr = np.vstack(node_degrees)
    mean_node_degrees = np.mean(node_degrees_arr, axis=0)
    node_degrees_idx = np.array([i for i in range(node_degrees_arr.shape[1])])
    error_node_degrees = np.std(node_degrees_arr, axis=0, ddof=1) / np.sqrt(node_degrees_arr.shape[1])

    grad, intercept = find_powerlaw_fit(node_degrees_idx, mean_node_degrees)
    mode_idx, mode_height = np.argmax(mean_node_degrees), np.max(mean_node_degrees)
    _maxent = nodeme.NodeME(k_limits=(3, 8), k_mean=3)
    maxent_pk = _maxent(target_pk=mode_height, k=mode_idx)

    ndfig, ndax = plt.subplots()
    ndax.errorbar(x=node_degrees_idx, y=mean_node_degrees, yerr=error_node_degrees, capsize=5.0)
    nd_smooth = np.linspace(0, node_degrees_arr.shape[0], 50)
    nd_powerlaw = np.exp(intercept) * np.linspace(0, node_degrees_arr.shape[0], 50)**(grad)
    nd_d, nd_p = stats.ks_2samp(nd_smooth, mean_node_degrees)
    ndax.plot(nd_smooth, nd_powerlaw, linestyle="dashed", color="black", label=f"s-f, p={nd_p:.2f}")
    if maxent_pk is not None:
        nd_maxent_d, nd_maxent_p = stats.ks_2samp(maxent_pk, mean_node_degrees)
        ndax.plot(_maxent.k, maxent_pk, linestyle="dotted", color="red", label=f"m-e, p={nd_maxent_p:.2f}")
    
    ndax.set_xlim(0, 8)
    ndax.set_ylim(0, 0.5)
    ndax.set_xlabel("node degree k")
    ndax.set_ylabel("p(r)")
    ndax.legend()
    filename = f"./Results/Node_Degrees/NODEDEGREES_{angle}_{length}.pdf"
    print("Saving fig to", filename)
    ndfig.savefig(filename)
    plt.close(ndfig) 

if __name__ == "__main__":

    REPETITIONS = range(1, 11)
    TOTAL_ASSORTATIVITIES = {}
    TOTAL_NONRANDOMNESS = {}
    TOTAL_SMETRICS = {}

    TOTAL_NODE_ASSORTATIVITIES = {}
    TOTAL_NODE_SMETRICS = {}
    for ANGLE in ["0.01", "0.05", "0.1", "0.5", "1", "5", "10", "100"]:
        for LENGTH in ["0.01", "0.05", "0.1", "0.5", "1", "5", "10", "100"]:
            PREFIX = f"NTMC_{ANGLE}_{LENGTH}"
            TOTAL_RDF = {i: [] for i in range(21)}
            TOTAL_GEOM_RDF = {i: [] for i in range(21)}
            NODE_ASSORTATIVITIES = []
            ASSORTATIVITIES = []
            NON_RANDOMNESS = []
            NODE_S_METRICS = []
            S_METRICS = []
            RING_SIZES = []
            NODE_DEGREES = []
            for REPETITION in REPETITIONS:
                DIRNAME = f"{PREFIX}_{REPETITION}"
                for SUBDIR in os.scandir(DIRNAME):
                    if not SUBDIR.is_dir():
                        continue
                    print(SUBDIR.path)
                    FILEPREFIX =  "./" + SUBDIR.path + f"/out_{ANGLE}_{LENGTH}_A"
                    DUALPREFIX =  "./" + SUBDIR.path + f"/out_{ANGLE}_{LENGTH}_B"
                    try:
                        POS_DICT, GRAPH, BOX = load_morley(FILEPREFIX)
                        DUAL_POS, DUAL_GRAPH, DUAL_BOX = load_morley(DUALPREFIX)
                    except FileNotFoundError:
                        print("Could not find file")
                        continue
                    try:
                        PRF = PeriodicRingFinder(graph=GRAPH, coords_dict=POS_DICT, cell=BOX[:, 1],
                                                 missing_policy="return")
                        DUAL_CNX = nx.get_node_attributes(DUAL_GRAPH, "dual_connections")
                        NEW_SHAPES = set()
                        ORIGINAL_COORDS = copy.deepcopy(POS_DICT)
                        for value in DUAL_CNX.values():
                            # Construct the edge list from the graph
                            start_node = min(value)
                            seen_nodes = set([start_node])
                            current_node = start_node
                            edges = []
                            while True:
                                neighbours = [item for item in GRAPH.neighbors(current_node)]
                                neighbours_in_ring = [neighbour for neighbour in neighbours if neighbour in value and neighbour not in seen_nodes]
                                if not neighbours_in_ring:
                                    if start_node in neighbours:
                                        edges.append(frozenset([current_node, start_node]))
                                    break
                                next_node = min(neighbours_in_ring)
                                seen_nodes.add(next_node)
                                edges.append(frozenset([current_node, next_node]))
                                current_node = next_node
                            shape = Shape(edges, coords_dict=POS_DICT, is_self_interacting=True)
                            NEW_SHAPES.add(shape)
                        PRF.current_rings = NEW_SHAPES
                        PRF.current_rings = PRF.add_ring_images(ORIGINAL_COORDS)
                        PRF.current_rings = PRF.find_unique_rings()
                    except RingFinderError as ex:
                        print("Ring finder error", ex)
                        continue
                    except nx.NetworkXError as ex:
                        print("NetworkXError", ex)
                        continue
                    FIG, AX = plt.subplots()
              
                    PRF.draw_onto(AX, cmap_name="coolwarm", min_ring_size=3)

                    RING_GRAPH = convert_to_ring_graph(PRF.current_rings)
                    RING_COLOURS = nx.greedy_color(RING_GRAPH, interchange=True)
                    nx.set_node_attributes(RING_GRAPH, RING_COLOURS, "color")
                    draw_periodic_coloured(RING_GRAPH,
                                           pos=nx.get_node_attributes(RING_GRAPH, "pos"),
                                           periodic_box=BOX,
                                           ax=AX,
                                           linestyle="dotted",
                                           )
                    print(f"Saving figure to ./Results/Networks/{ANGLE}_{LENGTH}_{REPETITION}.pdf")
                    FIG.savefig(f"./Results/Networks/{ANGLE}_{LENGTH}_{REPETITION}.pdf")
                    plt.close(FIG)
                    THIS_RING_SIZES = normalise_counter(Counter([len(ring) for ring in PRF.current_rings]))
                    
                    THIS_RING_SIZE_ARR = np.zeros(21)
                    for RS in THIS_RING_SIZES:
                        THIS_RING_SIZE_ARR[RS] = THIS_RING_SIZES[RS]   
                    RING_SIZES.append(THIS_RING_SIZE_ARR)

                    THIS_NODE_DEGREES = normalise_counter(Counter([degree for node_id, degree in GRAPH.degree()]))
                    THIS_NODE_DEGREES_ARR = np.zeros(9)
                    for ND in THIS_NODE_DEGREES:
                        THIS_NODE_DEGREES_ARR[ND] = THIS_NODE_DEGREES[ND]   
                    NODE_DEGREES.append(THIS_NODE_DEGREES_ARR)

                    MEAN_RDF, STD_RDF = topological_rdf(RING_GRAPH)
                    for ring_size in MEAN_RDF.keys():
                        TOTAL_RDF[ring_size].append(MEAN_RDF[ring_size])

                    MEAN_GEOM_RDF = geometric_rdf(RING_GRAPH, box=BOX[:, 1])
                    for ring_size in MEAN_GEOM_RDF.keys():
                        TOTAL_GEOM_RDF[ring_size].append(MEAN_GEOM_RDF[ring_size])
                    
                    ASSORTATIVITIES.append(nx.numeric_assortativity_coefficient(RING_GRAPH, "size"))
                    NODE_ASSORTATIVITIES.append(nx.degree_assortativity_coefficient(GRAPH))
                    try:
                        NON_RANDOMNESS.append(nx.non_randomness(RING_GRAPH))
                    except ValueError:
                        NON_RANDOMNESS.append((np.nan, np.nan))
                    gmax = li_smax_graph([item[1] for item in RING_GRAPH.degree()])
                    SMAX = nx.s_metric(gmax, normalized=False)
                    S_METRICS.append(nx.s_metric(RING_GRAPH, normalized=False) / SMAX)

                    node_gmax = li_smax_graph([item[1] for item in GRAPH.degree()])
                    NODE_SMAX = nx.s_metric(node_gmax, normalized=False)
                    NODE_S_METRICS.append(nx.s_metric(GRAPH, normalized=False) / NODE_SMAX)

            generate_ring_size_fig(RING_SIZES, ANGLE, LENGTH)
            generate_node_degree_fig(NODE_DEGREES, ANGLE, LENGTH)

            TOTAL_ASSORTATIVITIES[(LENGTH, ANGLE)] = np.mean(ASSORTATIVITIES), np.std(ASSORTATIVITIES, ddof=1)
            TOTAL_NONRANDOMNESS[(LENGTH, ANGLE)] = np.nanmean(NON_RANDOMNESS), np.nanstd(NON_RANDOMNESS, ddof=1)
            TOTAL_SMETRICS[(LENGTH, ANGLE)] = np.mean(S_METRICS), np.std(S_METRICS, ddof=1)

            TOTAL_NODE_ASSORTATIVITIES[(LENGTH, ANGLE)] = np.mean(NODE_ASSORTATIVITIES), np.std(NODE_ASSORTATIVITIES, ddof=1)
            TOTAL_NODE_SMETRICS[(LENGTH, ANGLE)] = np.mean(NODE_S_METRICS), np.std(NODE_S_METRICS, ddof=1)
            NEWFIG, NEWAX = plt.subplots()
            for RINGSIZE in TOTAL_RDF.keys():
                if RINGSIZE not in set([4, 6, 8, 10]):
                    continue
                if not TOTAL_RDF[RINGSIZE]:
                    continue
                max_data_size = max([len(item) for item in TOTAL_RDF[RINGSIZE]])
                for i, item in enumerate(TOTAL_RDF[RINGSIZE]):
                    if len(item) < max_data_size:
                        TOTAL_RDF[RINGSIZE][i] = np.pad(item, [0, max_data_size - len(item)])
                data = np.vstack(TOTAL_RDF[RINGSIZE])
                MEAN = np.mean(data, axis=0)
                STDEV = np.std(data, axis=0, ddof=1) / np.sqrt(len(REPETITIONS))
                NEWAX.plot([i for i in range(len(MEAN))], MEAN, label=f"{RINGSIZE}")
                NEWAX.fill_between([i for i in range(len(MEAN))], MEAN-STDEV, MEAN+STDEV, alpha=0.5)
            NEWAX.set_xlim(0, 10)
            NEWAX.set_xlabel("Topological Distance")
            NEWAX.plot([0, 10], [6, 6], linestyle="dashed", color="black", label="Average")
            NEWAX.set_ylim(0, 8)
            NEWAX.set_ylabel("Average ring size at distance")
            NEWAX.legend()
            NEWFIG.savefig(f"./Results/Topological_RDF/RDF_{ANGLE}_{LENGTH}.pdf")
            plt.close(NEWFIG)

            NEWFIG, NEWAX = plt.subplots()
            for RINGSIZE in TOTAL_GEOM_RDF.keys():
                if RINGSIZE not in set([4, 6, 8, 10]):
                    continue
                if not TOTAL_GEOM_RDF[RINGSIZE]:
                    continue
                max_data_size = max([len(item) for item in TOTAL_GEOM_RDF[RINGSIZE]])
                for i, item in enumerate(TOTAL_GEOM_RDF[RINGSIZE]):
                    if len(item) < max_data_size:
                        TOTAL_GEOM_RDF[RINGSIZE][i] = np.pad(item, [0, max_data_size - len(item)])
                data = np.vstack(TOTAL_GEOM_RDF[RINGSIZE])
                MEAN = np.mean(data, axis=0)
                STDEV = np.std(data, axis=0, ddof=1) / np.sqrt(len(REPETITIONS))
                NEWAX.plot([i for i in range(len(MEAN))], MEAN, label=f"{RINGSIZE}")
                NEWAX.fill_between([i for i in range(len(MEAN))], MEAN-STDEV, MEAN+STDEV, alpha=0.5)
            NEWAX.set_xlim(0, max(TOTAL_RDF.keys()))
            NEWAX.set_xlabel("Centroid Separation")
            NEWAX.plot([0, 20], [6, 6], linestyle="dashed", color="black", label="Average")
            NEWAX.set_ylim(0, 8)
            NEWAX.set_ylabel("Average ring size at distance")
            NEWAX.legend()
            NEWFIG.savefig(f"./Results/Geometric_RDF/GEOMRDF_{ANGLE}_{LENGTH}.pdf")
            plt.close(NEWFIG)

    with open('./Results/assortativities.dat', 'w') as fi:
        fi.write("Length, Angle, r, rstd\n")
        for key, val in TOTAL_ASSORTATIVITIES.items():
            fi.write(f"{key[0]}, {key[1]}, {val[0]:.3f}, {val[1]:.3f}\n")

    with open('./Results/smetrics.dat', 'w') as fi:
        fi.write("Length, Angle, s, sstd\n")
        for key, val in TOTAL_SMETRICS.items():
            fi.write(f"{key[0]}, {key[1]}, {val[0]:.3f}, {val[1]:.3f}\n")

    with open('./Results/assortativities.dat', 'w') as fi:
        fi.write("Length, Angle, r, rstd\n")
        for key, val in TOTAL_ASSORTATIVITIES.items():
            fi.write(f"{key[0]}, {key[1]}, {val[0]:.3f}, {val[1]:.3f}\n")

    with open('./Results/smetrics.dat', 'w') as fi:
        fi.write("Length, Angle, s, sstd\n")
        for key, val in TOTAL_SMETRICS.items():
            fi.write(f"{key[0]}, {key[1]}, {val[0]:.3f}, {val[1]:.3f}\n")

    with open('./Results/nonrandomness.dat', 'w') as fi:
        fi.write("Length, Angle, r, rstd\n")
        for key, val in TOTAL_NONRANDOMNESS.items():
            fi.write(f"{key[0]}, {key[1]}, {val[0]:.3f}, {val[1]:.3f}\n")

    
    
