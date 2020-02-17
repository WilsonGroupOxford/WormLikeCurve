#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:53:02 2019

@author: matthew-bailey
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.odr
import scipy.stats

import clustering


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


RVALUE_THRESHOLD = 0.8
MIN_LINE_LENGTH = 4

if __name__ == "__main__":
    COORDS = np.genfromtxt(
        "./Data/Results.csv", delimiter=",", skip_header=1, usecols=[4, 5], dtype=float
    )
    JUNC_TYPES = np.genfromtxt(
        "./Data/Results.csv", delimiter=",", skip_header=1, usecols=[7], dtype=str
    )

    CONTOUR_IDS = np.genfromtxt(
        "./Data/Results.csv", delimiter=",", skip_header=1, usecols=[2], dtype=int
    )
    FIG, AX = plt.subplots()
    BAD_LINES = []
    ALL_JUNCTIONS = []
    for CONTOUR_ID in np.unique(CONTOUR_IDS):
        MASK = CONTOUR_IDS == CONTOUR_ID
        POSITION_INDICIES = np.argwhere(CONTOUR_IDS == CONTOUR_ID)
        JUNCTION_POINTS = POSITION_INDICIES[[0, -1]]
        XS, YS = COORDS[:, 0][MASK], COORDS[:, 1][MASK]
        if len(XS) < MIN_LINE_LENGTH:
            continue
        SLOPE, INTERCEPT, RVALUE = fit_line(XS, YS)
        RVALUE = (1.0 - RVALUE) ** 2
        # We want this rvalue to be as close to zero as possible.
        if RVALUE > RVALUE_THRESHOLD:
            LINES = split_line(XS, YS)
        else:
            LINES = [(XS, YS)]

        # We've added new junction points by splitting up
        # the lines. Make a note of that.
        if len(LINES) > 2:
            OFFSETS = [line[0].shape for line in LINES]
            NEW_JUNCTIONS = [JUNCTION_POINTS + OFFSET for OFFSET in OFFSETS]
            JUNCTION_POINTS = np.vstack(JUNCTION_POINTS, *NEW_JUNCTIONS)
        for LINE in LINES:
            plot_line(XS, YS, AX, junction_points=[0, -1])
        ALL_JUNCTIONS.append(JUNCTION_POINTS)
    ALL_JUNCTIONS_FLAT = np.vstack(ALL_JUNCTIONS)[:, 0]
    LJ_PAIRS = clustering.find_lj_pairs(
        COORDS[ALL_JUNCTIONS_FLAT], ALL_JUNCTIONS_FLAT, 4.0
    )
    LJ_CLUSTERS = clustering.find_lj_clusters(LJ_PAIRS)
    CLUSTER_POSITIONS = clustering.find_cluster_centres(LJ_CLUSTERS, COORDS, offset=0)
    for key, value in CLUSTER_POSITIONS.items():
        CLUSTER_POSITIONS[key] = value * np.array([1.0, -1.0])
    MOLEC_TERMINALS = clustering.find_molecule_terminals(ALL_JUNCTIONS)
    G = nx.Graph()
    G = clustering.connect_clusters(G, MOLEC_TERMINALS, LJ_CLUSTERS)
    nx.draw(G, ax=AX, pos=CLUSTER_POSITIONS)
