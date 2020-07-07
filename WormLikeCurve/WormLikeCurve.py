#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:29:00 2019

@author: matthew-bailey
"""


import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

try:
    from WormLikeCurve.Bonds import AngleBond, HarmonicBond
    from WormLikeCurve.CurveCollection import CurveCollection
    from WormLikeCurve.Helpers import boltzmann_distrib
except ModuleNotFoundError:
    from Bonds import AngleBond, HarmonicBond
    from CurveCollection import CurveCollection
    from Helpers import boltzmann_distrib


class WormLikeCurve:
    """
    A class describing a Worm-like curve polymer.

    By default, it will generate a boltzmann distribution of segments. This
    can be sorted by overwriting the vectors array.
    The polymer is internally a collection of vectors,
    each described by a length and the angle of that vector
    with the positive x axis. Positions are calculated on
    the fly from this.
    """

    def __init__(
        self,
        harmonic_bond: HarmonicBond = HarmonicBond(k=1.0, length=50.0),
        angle_bond: AngleBond = AngleBond(k=100.0, angle=np.pi),
        num_segments: int = None,
        graph: nx.Graph = None,
        start_pos=np.array([0, 0]),
        kbt: float = 1.0,
    ):
        """
        Initialise the polymer object.

        :param num_segments: the number of segments of the polymer
        :param harmonic_bond: a two-body bond object
        representing the atom-atom bonds
        :param angle_bond: a three-body bond object
        representing the atom-atom-atom angle potential
        :param kbt: the combination of boltzmann constant * temp
        in the same unit system as is used for the bonds.
        """
        self.num_segments = num_segments
        self.harmonic_bond = harmonic_bond
        self.angle_bond = angle_bond
        self.kbt = kbt
        self.start_pos = start_pos.astype(float)
        self.bonds = []
        self.angles = []
        self._positions_dirty = True
        self._positions = None
        self._offset = np.zeros([2], dtype=float)
        self._circumcircle = None

        if num_segments is None and graph is None:
            raise RuntimeError("Must specify one of num_segments or graph")
        elif num_segments is not None and graph is not None:
            raise RuntimeError("Must specify only one of num_segments or graph")

        if num_segments is not None:
            graph = nx.path_graph(num_segments)
        # self.vectors = np.empty([len(self.bonds), 2])
        self._graph_to_vectors(graph)
        # Don't bother calculating the positions yet, do it
        # when they're necessary.

    def _graph_to_vectors(self, graph: nx.Graph):
        try:
            nx.find_cycle(graph)
            raise RuntimeError("Cannot turn a graph with cycles into a molecule")
        except nx.NetworkXNoCycle:
            # This is the desired state of affairs but networkx
            # is fussy, so just ignore and continue
            pass

        if not nx.is_connected(graph):
            raise RuntimeError("Graph must be connected to form one molecule")
        # Start from the smallest numbered singly-coordinate node
        single_nodes = [node for node, degree in graph.degree if degree == 1]
        start_node = min(single_nodes)

        # Set up the bonds and atom types
        self.bonds = [(None, None) for _ in range(len(graph.edges()))]
        self.vectors = np.zeros([len(graph.edges()), 2], dtype=float)
        # Assign the end nodes so there's a roughly equal number of type 2s
        # and type 3s across this molecule
        self.atom_types = np.ones([len(graph)], dtype=int)
        atom_types_counter = {2: 0, 3: 0}
        for single_node in single_nodes:
            min_atom_type = min(atom_types_counter, key=atom_types_counter.get)
            self.atom_types[single_node] = min_atom_type
            atom_types_counter[min_atom_type] += 1
        # Now assign the "branch" nodes
        for node in graph.nodes():
            if graph.degree(node) > 2:
                self.atom_types[node] = 4
        open_nodes = set([start_node])
        satisfied_nodes = set()
        self.angles = []
        bonds_counter = 0
        while open_nodes:
            this_node = open_nodes.pop()
            node_neighbours = sorted(list(nx.neighbors(graph, this_node)))
            num_neighbours = max(len(node_neighbours), 2)
            actual_neighbours = 1
            for neighbour in node_neighbours:
                if neighbour in satisfied_nodes:
                    continue
                self.bonds[bonds_counter] = (this_node, neighbour)
                length = self.generate_length()
                angle = (2 * actual_neighbours * np.pi / num_neighbours) - np.pi
                # This is a placeholder angle we update later
                self.vectors[bonds_counter, :] = length, angle
                # We now need to keep track of the cumulative angle up
                # until this point.
                shortest_path = nx.shortest_path(graph, start_node, neighbour)
                path_to_edges = [
                    (shortest_path[i], shortest_path[i + 1])
                    for i in range(len(shortest_path) - 1)
                ]
                edge_indices = [self.bonds.index(edge) for edge in path_to_edges]
                if this_node not in self.bonds[edge_indices[-1]]:
                    raise RuntimeError(
                        f"The shortest path from this neighbour to the start doesn't go through {this_node}"
                    )
                if len(edge_indices) >= 2:
                    last_edge = edge_indices[-2]
                    last_angle = self.vectors[last_edge, 1]
                else:
                    last_angle = 0.0
                self.vectors[bonds_counter, :] = length, angle + last_angle
                actual_neighbours += 1
                bonds_counter += 1

            open_nodes.update(node_neighbours)
            open_nodes = open_nodes.difference(satisfied_nodes)
            satisfied_nodes.add(this_node)
        self._positions_dirty = True
        self._positions = None

        # Generate the angles all the way around a node
        self.angles = []
        for node in sorted(graph.nodes()):
            if graph.degree(node) == 1:
                continue
            neighbours_angles = []
            for neighbour in graph.neighbors(node):
                vec = self.positions[neighbour] - self.positions[node]
                vec /= np.linalg.norm(vec)
                angle_with_x = np.arccos(vec[0])
                if vec[1] < 0:
                    angle_with_x *= -1
                neighbours_angles.append((neighbour, angle_with_x))
            # Now sort these according to their angle from the y axis
            # Adjacent entries in this list are "neighbouring vectors"
            sorted_angles = sorted(neighbours_angles, key=lambda tup: tup[1])

            # For threes and above, make sure the angles add up to 2pi.
            # don't do this for twos because that means we double-count
            if graph.degree(node) == 2:
                max_pair = len(sorted_angles) - 1
            else:
                max_pair = len(sorted_angles)

            for i in range(max_pair):
                if i == len(sorted_angles) - 1:
                    next_i = 0
                else:
                    next_i = i + 1
                pair = sorted_angles[i], sorted_angles[next_i]
                # remember stupid LAMMPS off-by-one
                self.angles.append(
                    (graph.degree(node) - 1, pair[1][0], node, pair[0][0])
                )

    @property
    def positions(self) -> np.array:
        """
        Return the Cartesian coordinates of the nodes in this polymer.

        This function is memoised, so will generate the positions afresh
        whenever they are "dirtied" by another function.
        """
        if self._positions_dirty:
            self._positions = self.vectors_to_positions()
        return self._positions

    def add_sticky_sites(self, proportion: float):
        """
        Turn a proportion of sites along the body (type 1) into 'sticky' sites (type 4).

        :param proportion: the fraction of sites between 0 and 1 to make sticky
        """
        # Fast past if we don't add any sticky sites.
        if proportion == 0:
            return
        assert 0 <= proportion, "Proportion must be a positive number between 0 and 1"
        assert 1 >= proportion, "Proportion must be a positive number between 0 and 1"
        num_body_atoms = self.num_atoms - 2
        body_atoms = np.random.choice(
            [1, 4], p=[1.0 - proportion, proportion], size=num_body_atoms
        )
        self.atom_types[1:-1] = body_atoms

    @property
    def centroid(self):
        """Return the location of the centre of mass of this curve."""
        # If we haven't calculated the positions, this is a bit hard.
        # Do it manually.
        return np.mean(self.positions, axis=0)

    def recentre(self):
        """Recentre the polymer so that its centre of mass is at start_pos."""
        self._offset = self.start_pos + self._offset - self.centroid
        self._positions_dirty = True
        self._positions = self.vectors_to_positions()

    def rotate(self, angle: float):
        """
        Rotate the polymer about its centroid.

        Does this
        by rotating the vector representation, and reconverting
        into Cartesians. Then recentres.
        :param angle: the angle to rotate about in radians.
        """
        self.vectors[:, 1] += angle
        self._positions_dirty = True
        self._circumcircle = None
        self.recentre()

    def rescale(self, scale_factor: float):
        """
        Rescale the size of this polymer by a given scale factor.

        Scales the starting position away from the origin, and makes
        all the vectors longer.

        :param scale_factor: the scale factor to change size by
        """
        # Change all the lengths by the scale factor
        self.vectors[:, 0] *= scale_factor
        # Move the origin, and regenerate.
        self.start_pos *= scale_factor
        self._offset *= scale_factor
        self._circumcircle = None
        self._positions_dirty = True

    def translate(self, translation: np.array):
        """
        Translate this polymer.

        :param translation: the vector to translate by.
        """
        # We can just move the starting position of this polymer
        # and then recalculate from the vectors.s
        self.start_pos += translation
        self._positions = self.vectors_to_positions()

    @property
    def num_atoms(self):
        """
        Return the number of atoms in the polymer.

        Because bonds link two atoms, the
        number of atoms is one more than the number of bonds.
        """
        return self._positions.shape[0]

    @property
    def num_angles(self):
        """
        Return the number of angle bonds in the polymer.

        Two bonds are linked by an angle,
        so the number of angles is one less than the number of bonds.
        """
        return len(self.angles)

    def generate_length(
        self, min_length: float = None, max_length: float = None, max_iters: int = 10
    ):
        """
        Generate a bond length for the harmonic bonds with a correct Boltzmann distribution.

        Randomly generates bond lengths between
        min_length and max_length, and calculates
        the energy. Makes a Monte Carlo-esque evaluation
        of the energy and rejects or accepts based on that.
        If the iterative process takes too long, just returns
        the mean length.
        :param min_length: minimum bond length, cannot be negative.
        Defaults to half the harmonic bond length if the argument is None
        :param max_length: minimum bond length, cannot be less than min_length.
        Defaults to twice the harmonic bond length if the argument is None.
        :param max_iters: how many iterations to try before returning the mean
        :return: length of a harmonic bond.
        """
        # Can't specify the lengths as an attribute in the signature,
        # so use None as a flag and set them to defaults here.
        if max_length is None:
            max_length = self.harmonic_bond.length * 2

        if min_length is None:
            min_length = self.harmonic_bond.length / 2

        if min_length < 0:
            raise ValueError("Minimum length must be positive")

        if max_length < min_length:
            raise ValueError(
                "Maximum length must be greater than or" + "equal to minimum length"
            )

        if max_iters < 0:
            raise ValueError("Maximum iterations must be a positive integer")
        iters_required = 0
        while True:
            iters_required += 1
            if iters_required > max_iters:
                return self.harmonic_bond.length
            length = np.random.uniform(min_length, max_length, 1)
            probability = boltzmann_distrib(self.harmonic_bond.energy(length), self.kbt)
            if probability > np.random.uniform():
                return length

    def generate_angle(
        self,
        angle_bond=None,
        min_angle: float = 0,
        max_angle: float = 2 * np.pi,
        max_iters: int = 10,
    ):
        """
        Generate a bond angle for the angular with a correct Boltzmann distribution.

        Randomly generates bond angles between
        min_angle and max_angle, and calculates
        the energy. Makes a Monte Carlo-esque evaluation
        of the energy and rejects or accepts based on that.
        If the iterative process takes too long, just returns
        the mean length.
        :param min_angle: minimum bond angle, default is 0.
        :param max_angle: maximum bond angle, default is 2pi.
        :param max_iters: how many iterations to try before returning the mean
        :return: angle between this vector and the previous one.
        """
        if angle_bond is None:
            angle_bond = self.angle_bond

        if min_angle < 0 or min_angle > 2 * np.pi:
            raise ValueError("Minimum angle must be in the range" + "0 to 2 pi.")

        if max_angle < min_angle:
            raise ValueError(
                "Maximum angle must be greater than or" + "equal to minimum angle"
            )

        if max_angle < 0 or max_angle > 2 * np.pi:
            raise ValueError("Minimum angle must be in the range" + "0 to 2 pi.")

        if max_iters < 0:
            raise ValueError("Maximum iterations must be a positive integer")

        iters_required = 0
        while True:
            iters_required += 1
            if iters_required > max_iters:
                return angle_bond.angle
            angle = np.random.uniform(min_angle, max_angle, 1)

            probability = boltzmann_distrib(angle_bond.energy(angle), self.kbt)
            if probability > np.random.uniform():
                return angle

    def positions_to_vectors(self):
        """Convert the [x, y] positself.vectors)ions back into vectors."""
        start_pos = self.positions[0, :]
        last_pos = start_pos
        vectors = []
        for position in self.positions[1:, :]:
            step = position - last_pos
            step_length = np.hypot(*step)
            normalised_step = step / step_length
            # The projection of this vector onto the x axis is
            # the x component
            angle_with_x = np.arccos(normalised_step[0])
            # Careful with quadrants!
            if normalised_step[1] < 0:
                angle_with_x = 2 * np.pi - angle_with_x
            vectors.append(np.array([step_length, angle_with_x]))
            last_pos = position
        # Finally, assign these to the class
        self.start_pos = start_pos
        self.vectors = np.vstack(vectors)
        return self.vectors

    def vectors_to_positions(self):
        """Convert the [r, theta] vectors into Cartesian coordinates."""
        positions = np.zeros([len(self.bonds) + 1, 2])
        start_node = self.bonds[0][0]
        positions[start_node, :] = self.start_pos + self._offset
        for i, bond in enumerate(self.bonds):
            length, angle = self.vectors[i]
            vector = np.array([length * np.cos(angle), length * np.sin(angle)])
            positions[bond[1]] = positions[bond[0]] + vector

        # Since we've recalculated, save these positions and mark
        # them as clean.
        self._positions = positions
        self._positions_dirty = False
        return self._positions

    def circumcircle_radius(self):
        if self._circumcircle is None:

            distance_from_origin = self.positions - self.centroid
            longest_distance = 0.0
            for row in distance_from_origin:
                distance = np.linalg.norm(row)
                longest_distance = max(0.0, distance)
            self._circumcircle = longest_distance
        return self._circumcircle

    def plot_onto(
        self, ax, fit_edges: bool = True, label_nodes=False, index_offset=0, **kwargs
    ):
        """
        Plot this polymer as a collection of lines, detailed by kwargs into the provided axis.

        :param ax: a matplotlib axis object to plot onto
        :param fit_edges: whether to resize xlim and ylim to fit this polymer.
        """

        END_SIZE = kwargs.pop("end_size", 10)
        lines = []
        for bond in self.bonds:
            lines.append(
                np.row_stack([self.positions[bond[0]], self.positions[bond[1]]])
            )
        collection_linewidths = kwargs.pop("linewidths", 5)
        collection_colours = kwargs.pop("colors", "purple")
        behind_line_collection = LineCollection(
            lines, linewidths=collection_linewidths + 3, colors="black", **kwargs
        )
        ax.add_collection(behind_line_collection)
        line_collection = LineCollection(
            lines, linewidths=collection_linewidths, colors=collection_colours, **kwargs
        )
        ax.add_collection(line_collection)

        # None at the start for stupid LAMMPS off-by-one
        TYPES_TO_COLOURS = [None, "purple", "blue", "green", "red"]
        # Now draw the sticky ends
        for i in range(self.positions.shape[0]):
            circle_end = mpatches.Circle(
                self.positions[i],
                END_SIZE / 2,
                facecolor=TYPES_TO_COLOURS[self.atom_types[i]],
                edgecolor="black",
                linewidth=3,
                zorder=3,
                **kwargs,
            )
            ax.add_artist(circle_end)
        if label_nodes:
            for index in range(self.positions.shape[0]):
                text = ax.text(
                    self.positions[index, 0],
                    self.positions[index, 1],
                    index + index_offset,
                    color=TYPES_TO_COLOURS[self.atom_types[index]],
                    zorder=4,
                    ha="center",
                    va="center",
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=3, foreground="black"),
                        path_effects.Normal(),
                    ]
                )
        if fit_edges:
            min_x, min_y = np.min(self.positions, axis=0)
            max_x, max_y = np.max(self.positions, axis=0)
            min_corner = (min(min_x, min_y) * 1.1) - 0.1
            max_corner = (max(max_x, max_y) * 1.1) + 0.1
            ax.set_xlim(min_corner, max_corner)
            ax.set_ylim(min_corner, max_corner)

    def to_lammps(self, filename: str, *args, **kwargs):
        """
        Write out to a lammps file which can be read by a read_data command.

        :param filename: the name of the file to write to.
        """
        CurveCollection(self).to_lammps(filename, *args, **kwargs)

    def apply_transformation_matrix(self, matrix: np.array):
        """
        Apply an arbitrary transformation to the polymer.

        This transformation is a 2x2 matrix with real
        components that represents a 2D scale, skew etc.
        Rotation and isotropic scaling matrices are specialised in
        rotate() and rescale() functions with some safety checks.
        This does not recentre on the origin for rotations,
        so best to use rotate() for that.

        :param matrix: the 2x2 transformation matrix
        """
        for i, position in enumerate(self.positions):
            self.positions[i, :] = np.matmul(matrix, position)
        # Keep the vector representation in line with the
        # cartesian representation.
        self.positions_to_vectors()
        return self

    def collides_with(self, other, periodic_box=None) -> bool:
        """
        Test if this WLC collides with another WLC.
        
        A collision is defined if two atoms in different molecules are within half a
        bond length of one another. 
        :param other: the other wormlike curve to test
        :param periodic_box: A box to apply minimum image conventions over. Can be None, for no periodicity. Should be in the form [[x_min, x_max], [y_min, y_max], [z_min, z_max]] or [[x_min, x_max], [y_min, y_max]].
        """

        # Ultra fast path -- this is us, so we can't collide with ourselves!
        if self is other:
            return False
        # Fast path, are these circumcircles within a cutoff.
        self_bond_length = self.harmonic_bond.length / 2
        self_circumcircle = self.circumcircle_radius()
        other_bond_length = other.harmonic_bond.length / 2
        other_circumcircle = self.circumcircle_radius()

        centroid_separation = np.linalg.norm(other.centroid - self.centroid)
        if (
            centroid_separation
            > self_circumcircle
            + other_circumcircle
            + self_bond_length
            + other_bond_length
        ):
            return False

        # Slow path, now we have to check all atoms.
        for position in self.positions:
            for other_position in other.positions:
                separation = position - other_position
                # Minimum image convention correction.
                if periodic_box is not None:
                    for dim in range(separation.shape[0]):
                        halfbox = np.abs(periodic_box[dim, 1] - periodic_box[dim, 0])
                        if separation[dim] > halfbox:
                            separation[dim] -= halfbox
                        elif separation[dim] < -halfbox:
                            separation[dim] += halfbox
                distance = np.linalg.norm(separation)

                if distance < self_bond_length + other_bond_length:
                    # We've detected a collision!
                    return True
        # No collisions found, we're in the clear.
        return False


if __name__ == "__main__":
    TRIANGLE_GRAPH = nx.Graph()
    TRIANGLE_GRAPH.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9)]
    )

    # nx.draw(TRIANGLE_GRAPH)

    DOUBLE_TRIANGLE_GRAPH = nx.Graph()
    DOUBLE_TRIANGLE_GRAPH.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (9, 13),
            (13, 14),
            (14, 15),
        ]
    )
    TEST_GRAPH = nx.Graph()
    TEST_GRAPH.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6)])
    WLC = WormLikeCurve(
        graph=DOUBLE_TRIANGLE_GRAPH,
        harmonic_bond=HarmonicBond(1.0, 50.0),
        angle_bond=AngleBond(1.0, 1.0),
        start_pos=np.array([0.0, 0.0]),
    )
    print(WLC.angles)
    WLC.recentre()
    WLC.to_lammps("./test.data", mass=0.07)
    FIG, AX = plt.subplots()
    WLC.plot_onto(AX, label_nodes=True)
    FIG.show()
