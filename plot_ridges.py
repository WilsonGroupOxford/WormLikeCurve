#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:53:02 2019

@author: matthew-bailey
"""

import sys
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.odr
import scipy.stats
from PIL import Image, ImageOps

import clustering
import morley_parser
from rings.ring_finder import RingFinder, convert_to_ring_graph


def group_lines(lines_list, distance_threshold, angle_threshold):
    lines_to_remove = set()
    lines_to_add = list()
    for i in range(len(lines_list)):
        line = lines_list[i]
        if i in lines_to_remove:
            # Don't do anything with this line if
            # it's been grouped once.
            continue
        for j in range(i):
            other_line = lines_list[j]
            if j in lines_to_remove:
                # Don't do anything with this line if
                # it's been grouped once.
                continue
            junction_distances = [
                other_line.distance_from(line.coordinates[junc])
                for junc in line.junction_points
            ]
            min_distance = min(junction_distances)
            if min_distance < distance_threshold:
                # Now assess the slope
                angle_between = np.arctan(
                    (line.slope - other_line.slope)
                    / (1 + line.slope * other_line.slope)
                )
                if np.abs(angle_between) < angle_threshold:
                    # We need to merge these two lines
                    new_coords = np.concatenate(
                        [line.coordinates, other_line.coordinates], axis=0
                    )
                    new_line = Line(new_coords)
                    lines_to_add.append(new_line)
                    lines_to_remove.update([i, j])
    for idx_to_remove in sorted(lines_to_remove, reverse=True):
        del lines_list[idx_to_remove]
    lines_list.extend(lines_to_add)
    return lines_list


def group_lines_graph(in_graph, positions, max_length, angle_cutoff=np.pi / 8):
    changed = True
    while changed:
        changed = False
        edges_to_remove = set()
        edges_to_add = set()
        for node in in_graph.nodes():
            neighbours = tuple(in_graph.neighbors(node))
            print(tuple(neighbours))
            if len(neighbours) != 2:
                # Skip all the ones that aren't 2-coordinate because
                # we'll screw up the linking otherwise.
                continue
            # Check the size of the new line that we might make. If it's too long,
            # don't carry on.
            distance_a_b = np.linalg.norm(
                positions[neighbours[1]] - positions[neighbours[0]]
            )
            if distance_a_b > max_length:
                continue

            vec_a = positions[neighbours[0]] - positions[node]
            vec_a /= np.linalg.norm(vec_a)

            vec_b = positions[neighbours[1]] - positions[node]
            vec_b /= np.linalg.norm(vec_b)

            angle_between = np.arccos(np.dot(vec_a, vec_b))
            if np.abs(angle_between - np.pi) < angle_cutoff:
                print("found a node to remove:", node)
                edges_to_remove.add((neighbours[0], node))
                edges_to_remove.add((neighbours[1], node))

                edges_to_add.add(frozenset([neighbours[0], neighbours[1]]))
                break
        if edges_to_remove:
            changed = True
            print("Adding edges", [tuple(item) for item in edges_to_add])
            print("Removing edges", [tuple(item) for item in edges_to_remove])
            in_graph.add_edges_from(edges_to_add)
            in_graph.remove_edges_from(edges_to_remove)


def fit_line(x_values, y_values):
    """
    Fit an orthogonal least squares line to the
    set of points passed in. Should handle vertical lines
    well, although the first guess might be a little iffy.
    :param x_values: an Nx1 array of x coordinates
    :param y_values: an Nx1 array of y coordinates
    :return slope: the gradient of the fitted line
    :return intercept: the intercept of the fitted line
    :return res_var: the residual variance, as a difference
    from 1. Lower is better.
    """
    data = scipy.odr.Data(x_values, y_values)
    model = scipy.odr.unilinear

    # Perform a first guess to help the ODR out.
    _slope, _intercept, _, _, _ = scipy.stats.linregress(x_values, y_values)
    ODRResult = scipy.odr.ODR(data, model, beta0=[_slope, _intercept]).run()
    res_var = ODRResult.res_var
    # These cause floating point overflows if our model is terrible
    # so make them infinite.
    if res_var > 1e150:
        res_var = float("Inf")
    elif res_var < -1e150:
        res_var = float("-Inf")
    return ODRResult.beta[0], ODRResult.beta[1], (1.0 - res_var) ** 2


def find_nearest_points(x_values, y_values, slope: float, intercept: float):
    """
    Given a set of x and y coordinates, and a line, returns the
    x and y coordinates that are on the line and closest to a given
    point.
    :param x_values: an Nx1 array of x values corresponding to points
    :param y_values: an Nx1 array of y values corresponding to points
    :return nearest_xs: the x coordinates of the points on the line
    closest to the input coordinates
    :return nearest_ys: the y coordinates of the points on the line
    closest to the input coordinates
    """
    nearest_xs = ((x_values + slope * y_values) - slope * intercept) / (slope ** 2 + 1)
    nearest_ys = (slope * (x_values + slope * y_values) + intercept) / (slope ** 2 + 1)
    return nearest_xs, nearest_ys


def plot_line(xs, ys, ax, junction_points=None, plot_fit: bool = False):
    """
    Plots a set of points onto an axis consistently.
    If junction points are specified, will plot the
    junction points in a different style.
    If plot_fit is true, will plot the line of best
    fit as well.
    :param xs: an Nx1 array of x coordinates.
    :param ys: an Nx1 array of y coordinates.
    :param junction_points: a list of indicies which correspond to
    junctions. Can be None, indicating no junctions.
    :param plot_fit: plot the line of best fit as well.
    """
    this_plot = ax.scatter(xs, -ys, marker="x",)
    if plot_fit:
        slope, intercept, rvalue = fit_line(xs, ys)
        nearest_xs, nearest_ys = find_nearest_points(xs, -ys, slope, intercept)
        ax.scatter(nearest_xs, -nearest_ys, color=this_plot.get_edgecolor())

    if junction_points is not None:
        ax.scatter(
            xs[junction_points],
            -ys[junction_points],
            marker="s",
            color=this_plot.get_edgecolor(),
            s=80,
        )


def split_line(xs, ys):
    """
    Attempts to split a set of data with a poor line
    of best fit into two better lines of best fit.
    :param xs: an Nx1 array of x coordinates.
    :param ys: an Nx1 array of y coordinates.
    :return split_lines: an iterable containing the
    split lines. Returns an iterable just containing
    the line if the splits are all worse.
    """
    # TODO: We recalculate the fit every time. Could this
    # be sped up?
    _, _, rvalue = fit_line(xs, ys)
    split_rvalues = []
    slopes = []
    intercepts = []
    # Test all possible splitting points.
    for midpoint in range(2, np.shape(xs)[0] - 1):
        these_rvalues = []
        these_slopes = []
        these_intercepts = []
        # TODO: Could make this recursive, and split down to a minimum
        # size?
        for split_xs, split_ys in [
            (xs[midpoint:], ys[midpoint:]),
            (xs[:midpoint], ys[:midpoint]),
        ]:
            split_slope, split_intercept, res_var = fit_line(split_xs, split_ys)
            these_rvalues.append(res_var)
            these_slopes.append(split_slope)
            these_intercepts.append(split_intercept)
        # TODO: This is an extremely dodgy metric.
        # Should calculate a proper residual error metric.
        split_rvalues.append(np.mean(these_rvalues))
        slopes.append(these_slopes)
        intercepts.append(these_intercepts)

    # If any of these are better than
    if min(split_rvalues) < rvalue:
        split_index = split_rvalues.index(min(split_rvalues))
        return [
            (xs[split_index:], ys[split_index:]),
            (xs[:split_index], ys[:split_index]),
        ]
    return [(xs, ys)]


def analyse_node_degree(real_graph):
    """
    Analyse the node degree distribution, returning the mean and standard deviation.
    """
    node_degrees = np.array([degree for node_id, degree in real_graph.degree()])
    mean_node_degree = np.mean(node_degrees)
    std_node_degree = np.std(node_degrees, ddof=1)
    return mean_node_degree, std_node_degree


def node_degree_histogram(real_graph, normalise=True):
    """
    Analyse the node degree distribution, returning the mean and standard deviation.
    """
    node_degrees = np.array([degree for node_id, degree in real_graph.degree()])
    node_degree_hist = np.empty(np.max(node_degrees) + 1, dtype=float)
    for item in node_degrees:
        node_degree_hist[item] += 1
    if normalise:
        node_degree_hist /= np.sum(node_degree_hist)
    return node_degree_hist


RVALUE_THRESHOLD = 0.8
MIN_LINE_LENGTH = 4


class Line:
    def __init__(self, coordinates, junctions=None):
        self.coordinates = coordinates
        if junctions is not None:
            self.junction_points = junctions
        else:
            min_y_idx = np.argmin(self.coordinates[:, 1])
            max_y_idx = np.argmax(self.coordinates[:, 1])

            y_diff = self.coordinates[max_y_idx, 1] - self.coordinates[min_y_idx, 1]

            min_x_idx = np.argmin(self.coordinates[:, 0])
            max_x_idx = np.argmax(self.coordinates[:, 0])

            x_diff = self.coordinates[max_x_idx, 0] - self.coordinates[min_x_idx, 0]

            if y_diff > x_diff:
                self.junction_points = [min_y_idx, max_y_idx]
            else:
                self.junction_points = [min_x_idx, max_x_idx]

        self.slope, self.intercept, self.rvalue = fit_line(
            self.coordinates[:, 0], self.coordinates[:, 1]
        )

    def split(self, test_points=(0.25, 0.5, 0.75), rvalue_policy="geometric"):

        r_values = np.zeros([2, len(test_points)])
        tested_lines = []
        for i, test_point in enumerate(test_points):
            split_index = int(len(self.coordinates) * test_point)
            # The line doesn't have enough data to split it here.
            if split_index in {0, 1, len(self.coordinates) - 1, len(self.coordinates)}:
                tested_lines.append((self,))
                r_values[:, i] = -np.inf, -np.inf
                continue

            line_a = Line(self.coordinates[:split_index, :])
            line_b = Line(self.coordinates[split_index:, :])
            tested_lines.append((line_a, line_b))

            r_values[:, i] = line_a.rvalue, line_b.rvalue

        if rvalue_policy == "geometric":
            new_r_values = np.sqrt(np.prod(r_values, axis=0))
        elif rvalue_policy == "mean" or rvalue_policy == "arithmetic":
            new_r_values = np.mean(r_values, axis=0)
        elif rvalue_policy == "max":
            new_r_values = np.max(r_values, axis=0)
        else:
            raise RuntimeError(
                "Bad rvalue policy, must be geometric, mean, arithemetic or max"
            )
        max_r_idx = np.argmax(new_r_values)
        if new_r_values[max_r_idx] > self.rvalue:
            return tested_lines[max_r_idx]
        return [self]

    def __str__(self):
        return f"{self.coordinates[self.junction_points[0]]} -> {self.coordinates[self.junction_points[-1]]}"

    def plot_onto(self, ax=None, plot_fit=False):
        plot_line(
            ax=ax,
            xs=self.coordinates[:, 0],
            ys=self.coordinates[:, 1],
            plot_fit=plot_fit,
        )

    def distance_from(self, point):
        """
        Calculate the distance from a point to this line segment.

        If the projection of the point onto the line is within the line,
        this is the perpendicular distance. If the projection is outside the line,
        it's the euclidean distance to the nearest end.
        """
        if self.projection_within_line(point):
            start_point = self.coordinates[self.junction_points[0]]
            end_point = self.coordinates[self.junction_points[1]]
            step = end_point - start_point
            normed_step = step / np.linalg.norm(step)
            start_to_point = point - start_point
            print(start_to_point, normed_step, np.dot(start_to_point, normed_step))
            return np.linalg.norm(
                start_to_point - np.dot(start_to_point, normed_step) * normed_step
            )
        distance_to_edges = [
            np.linalg.norm(point - self.coordinates[self.junction_points[0]]),
            np.linalg.norm(point - self.coordinates[self.junction_points[1]]),
        ]
        return min(distance_to_edges)

    def projection_within_line(self, other_point):
        start_point = self.coordinates[self.junction_points[0]]
        end_point = self.coordinates[self.junction_points[1]]
        step = end_point - start_point
        start_to_point = other_point - start_point
        overlap = np.dot(step, start_to_point) / np.dot(step, step)
        if 0.0 <= overlap and 1.0 >= overlap:
            return True
        return False

    def test_intersect(self, other):
        def orientation(point_a, point_b, point_c):
            discrim = (point_b[1] - point_a[1]) * (point_c[0] - point_b[0]) - (
                point_c[1] - point_b[1]
            ) * (point_b[0] - point_a[0])
            if discrim < 0:
                return -1
            if discrim > 0:
                return 1
            return 0

        p_1 = self.coordinates[self.junction_points[0]]
        q_1 = self.coordinates[self.junction_points[-1]]

        p_2 = other.coordinates[other.junction_points[0]]
        q_2 = other.coordinates[other.junction_points[-1]]

        orientations = [
            orientation(p_1, q_1, p_2),
            orientation(p_1, q_1, q_2),
            orientation(p_2, q_2, p_1),
            orientation(p_2, q_2, q_1),
        ]
        if (orientations[0] * orientations[1] == -1) and (
            orientations[2] * orientations[3] == -1
        ):
            return True

        # all collinear case
        if not any(orientations):
            print("Collinear case")
            p_1_x = [p_1[0], 0.0]
            q_1_x = [q_1[0], 0.0]
            p_2_x = [p_2[0], 0.0]
            q_2_x = [q_2[0], 0.0]
            orientations_x = [
                orientation(p_1_x, q_1_x, p_2_x),
                orientation(p_1_x, q_1_x, q_2_x),
                orientation(p_2_x, q_2_x, p_1_x),
                orientation(p_2_x, q_2_x, q_1_x),
            ]
            print("Orientations_x", orientations_x, p_1_x, q_1_x, p_2_x, q_2_x)
            if (orientations_x[0] * orientations_x[1] == -1) and (
                orientations_x[2] * orientations_x[3] == -1
            ):
                return True

            p_1_y = [0.0, p_1[1]]
            q_1_y = [0.0, q_1[1]]
            p_2_y = [0.0, p_2[1]]
            q_2_y = [0.0, q_2[1]]
            orientations_y = [
                orientation(p_1_y, q_1_y, p_2_y),
                orientation(p_1_y, q_1_y, q_2_y),
                orientation(p_2_y, q_2_y, p_1_y),
                orientation(p_2_y, q_2_y, q_1_y),
            ]
            print("Orientations_y", orientations_y)
            if (orientations_y[0] * orientations_y[1] == -1) and (
                orientations_y[2] * orientations_y[3] == -1
            ):
                return True
        return False


if __name__ == "__main__":
    if len(sys.argv) == 2:
        FILENAME = sys.argv[1]
        BASENAME = os.path.splitext(os.path.basename(FILENAME))[0]
        print(BASENAME)
    else:
        FILENAME = "./Data/Results.csv"
        BASENAME = "Results"
    print("Analysing", FILENAME)
    COORDS = np.genfromtxt(
        FILENAME, delimiter=",", skip_header=1, usecols=[4, 5], dtype=float
    ) * np.array([1.0, -1.0])
    RANGE = np.max(COORDS) - np.min(COORDS)
    JUNC_TYPES = np.genfromtxt(
        FILENAME, delimiter=",", skip_header=1, usecols=[7], dtype=str
    )

    CONTOUR_IDS = np.genfromtxt(
        FILENAME, delimiter=",", skip_header=1, usecols=[2], dtype=int
    )
    FIG, AX = plt.subplots()
    EDGES = set()
    ALL_LINES = []
    for CONTOUR_ID in np.unique(CONTOUR_IDS):
        MASK = CONTOUR_IDS == CONTOUR_ID
        POSITION_INDICIES = np.argwhere(CONTOUR_IDS == CONTOUR_ID)
        LINE = Line(COORDS[MASK])
        if LINE.rvalue < RVALUE_THRESHOLD:
            NEW_LINES = LINE.split()
        else:
            NEW_LINES = [LINE]
        ALL_LINES.extend(NEW_LINES)

    JUNCTION_POS = np.array(
        [line.coordinates[junc] for line in ALL_LINES for junc in line.junction_points]
    )
    JUNCTION_IDS = [i for i in range(JUNCTION_POS.shape[0])]
    for i in range(0, len(JUNCTION_IDS), 2):
        EDGES.add((JUNCTION_IDS[i], JUNCTION_IDS[i + 1]))

    LJ_PAIRS = clustering.find_lj_pairs(JUNCTION_POS, JUNCTION_IDS, 0.005 * RANGE)
    LJ_CLUSTERS = clustering.find_lj_clusters(LJ_PAIRS)

    CLUSTER_POSITIONS = clustering.find_cluster_centres(
        LJ_CLUSTERS, JUNCTION_POS, offset=0
    )
    # for key, value in CLUSTER_POSITIONS.items():
    #    CLUSTER_POSITIONS[key] = value * np.array([1.0, -1.0])

    # MOLEC_TERMINALS = clustering.find_molecule_terminals(ALL_JUNCTIONS)
    G = nx.Graph()
    G.add_edges_from(EDGES)
    nx.set_node_attributes(G, 2, "atom_types")
    out_graph = clustering.connect_clusters(
        in_graph=G, clusters=sorted(list(LJ_CLUSTERS))
    )
    # group_lines_graph(out_graph, CLUSTER_POSITIONS, 0.5*RANGE, np.pi/8)
    morley_parser.colour_graph(out_graph)

    rf = RingFinder(out_graph, CLUSTER_POSITIONS)
    rf.draw_onto(AX, cmap_name="viridis", alpha=0.5)
    morley_parser.draw_nonperiodic_coloured(out_graph, ax=AX, pos=CLUSTER_POSITIONS)
    FIG.savefig(f"./Processed/{BASENAME}_rings.pdf", bbox_inches="tight")
    if BASENAME is not None:
        # base_im = Image.open(f"./Images/{BASENAME}.png")
        # AX.imshow(base_im, origin="upper")
        FIG.savefig(f"./Processed/{BASENAME}_overlay.pdf", bbox_inches="tight")

    # Now do the analysis.
    ND_MEAN, ND_STD = analyse_node_degree(out_graph)
    ND_HIST = node_degree_histogram(out_graph)
    NODE_ASSORT = nx.degree_assortativity_coefficient(out_graph)

    RING_GRAPH = convert_to_ring_graph(rf.current_rings)
    RING_ASSORT = nx.numeric_assortativity_coefficient(RING_GRAPH, "size")
    RING_SIZES = np.array([len(ring) for ring in rf.current_rings])
    MEAN_RING_SIZE = np.mean(RING_SIZES)
    STD_RING_SIZE = np.std(RING_SIZES, ddof=1)

    with open(f"./Processed/{BASENAME}_results.txt", "w") as fi:
        fi.write(f"Mean Degree, {ND_MEAN}\n")
        fi.write(f"Std Degree , {ND_STD}\n")
        fi.write(f"Node r     , {NODE_ASSORT}\n")
        fi.write(f"Mean Ring  , {MEAN_RING_SIZE}\n")
        fi.write(f"Std Ring   , {STD_RING_SIZE}\n")
        fi.write(f"Ring r     , {RING_ASSORT}\n")
    # plt.show()
