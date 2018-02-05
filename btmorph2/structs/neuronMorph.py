from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import object
from past.utils import old_div
from .basic import Tree
from ..btviz import plot_3D, animate, plot_dendrogram, plot_2D
import numpy as np
import sys
from .defaults import defaultGlobalScalarFuncs
from ..auxFuncs import getIntersectionXYZs

class NeuronMorphology(object):
    '''
    Neuron for use with a Tree (:class:`Tree`).

    Essentially a wrapper for Tree Class, represents a neuron object for which
    statistical analysis can be applied to
    '''

    def __init__(self, input_file=None, pca_translate=False,
                 translate_origin=None, width="x", height="z",
                 depth="y", correctIfSomaAbsent=False,
                 ignore_type=False):

        """
        Default constructor.

        Parameters
        -----------
        input_file : :class:`str`
            File name of neuron to be created
        pca_translate : boolean
            Default False. If True, the morphology is translated along the
                first 3 axes of the PCA performed on its compartments. The
                translated morphology is stored and all future references
                (morphometric and visualization) will use the translated
                morphology
        translate_origin : array of floats
            Default None. If set, this is a 3D array to define the location
                of the soma. Format is ['x', 'z', 'y'] The translated
                morphology will be stored. If used in conjunction with the
                pca_translate, the morphology will first be translated
                according to PCA and then moved to the specified soma location
        width : string
            Either "x", "y" or "z" to determine which axis in the SWC format
                corresponds to the internally stored axis.
        height : string
            Either "x", "y" or "z" to determine which axis in the SWC format
                corresponds to the internally stored axis.
        depth : string
            Either "x", "y" or "z" to determine which axis in the SWC format
                corresponds to the internally stored axis.
        correctIfSomaAbsent: bool
            if True, then for trees whose roots are not of type 1, the roots are
            manually set to be of type 1 and treated as they have one point soma.
        ignore_type: bool
            if True, the 'type' value in the second column are ignored
        """

        axis_config = [0, 0, 0]
        if width is "x":
            axis_config[0] = 0
        elif width is "z":
            axis_config[0] = 1
        elif width is "y":
            axis_config[0] = 2
        if height is "x":
            axis_config[1] = 0
        elif height is "z":
            axis_config[1] = 1
        elif height is "y":
            axis_config[1] = 2
        if depth is "x":
            axis_config[2] = 0
        elif depth is "z":
            axis_config[2] = 1
        elif depth is "y":
            axis_config[2] = 2
        if sum(axis_config) is not 3:
            raise Exception("Axes incorrectly set, \
                             ensure each axis is set correctly")

        if input_file is not None:
            self.axis_config = axis_config
            self.correctIfSomaAbsent = correctIfSomaAbsent
            self.ignore_type = ignore_type
            self.file = input_file

        if pca_translate:
            self.get_tree().pca_project_tree()

        if translate_origin is not None:
            rootPos = self.get_tree().get_root().content['p3d'].xyz
            if rootPos[0] != translate_origin[0] and \
                            rootPos[1] != translate_origin[1] and \
                            rootPos[2] != translate_origin[2]:

                translate = translate_origin - rootPos
                for n in self.get_tree().get_nodes():
                    n.content['p3d'].xyz = n.content['p3d'].xyz + translate

    def set_file(self, input_file):
        """
        Set the SWC file

        Parameters
        -----------
        input_file : :class:`str`
            File name of neuron to be created,
        """
        self.__file = input_file
        self.tree = None

    def get_file(self):

        """
        Obtain the root Node

        Returns
        -------
        root : :class:`Node`
        """
        return self.__file

    file = property(get_file, set_file)

    def set_tree(self, tree=None):

        """
        Set the SWC file

        Parameters
        -----------
        input_file : :class:`str`
            File name of neuron to be created,
        """
        if tree is None:
            self.__tree = Tree(self.file, self.axis_config,
                               self.correctIfSomaAbsent, self.ignore_type)
        else:
            self.__tree = tree
        self._all_nodes = self.tree.get_nodes()

        # compute some of the most used stats +
        self._soma_points, self._bif_points, self._end_points = \
            self.get_points_of_interest()

    def get_tree(self):

        """
        Obtain the root Node

        Returns
        -------
        root : :class:`Node`
        """
        return self.__tree

    tree = property(get_tree, set_tree)

    def plot_3DGL(self, displaysize=(800, 600), zoom=10, poly=True,
                  fast=False, multisample=True, graph=True):

        """
        Gate way to btvizGL3D plot

        Parameters
        -----------
        displaysize : tuple (int, int)
            set size of window, default is 800,600
        zoom : int
            distance from centre to start from
        poly : boolean
            Draw Polygon or Wire frame representation
        fast : boolean
            Increase fps by removing overlapping branches (only if poly = true)
        multisample : boolean
            Improves quality of image through multisample (false improves fps)
        graph : boolean
            Start with Graph enabled (can toggle while running with 'g' key)
        """
        from btmorph2.btvizGL import btvizGL

        window = btvizGL()
        window.Plot(self, displaysize, zoom, poly, fast, multisample, graph)
        window = None

    def animateGL(self, filename, displaysize=(800, 600), zoom=5,
                  poly=True, axis='z', graph=False):

        """
        Gate way to btvizGL

        Parameters
        -----------
        filename : string
            Filename for gif produces (available in captures/animations folder)
        displaysize : tuple (int, int)
            set size of window, default is 800,600
        zoom : int
            distance from centre to start from
        poly : boolean
            Draw Polygon or Wire frame representation
        graph : boolean
            Start with Graph enabled (can toggle while running with 'g' key)

        """
        from btmorph2.btvizGL import btvizGL

        window = btvizGL()
        window.Animate(self, filename, displaysize, zoom, poly, axis, graph)
        window = None

    def plot_3D(self, color_scheme="default", color_mapping=None,
                synapses=None, save_image=None, show_radius=True):

        """
        Gate way to btviz plot_3D_SWC to create object orientated relationship

        3D matplotlib plot of a neuronal morphology. The SWC has to be
        formatted with a "three point soma".
        Colors can be provided and synapse location marked

        Parameters
        ----------
        color_scheme: string
            "default" or "neuromorpho". "neuronmorpho" is high contrast
        color_mapping: list[float] or list[list[float,float,float]]
            Default is None. If present, this is a list[N] of colors
            where N is the number of compartments, which roughly corresponds to
            the number of lines in the SWC file. If in format of list[float],
            this list is normalized and mapped to the jet color map, if in
            format of list[list[float,float,float,float]], the 4 floats represt
            R,G,B,A respectively and must be between 0-255. When not None, this
            argument overrides the color_scheme argument(Note the difference
            with segments).
        synapses : vector of bools
            Default is None. If present, draw a circle or dot in a distinct
            color at the location of the corresponding compartment. This is
            a 1xN vector.
        save_image: string
            Default is None. If present, should be in format
            "file_name.extension", and figure produced will be saved as
            this filename.
        show_radius : boolean
            True (default) to plot the actual radius. If set to False,
            the radius will be taken from `btmorph2\config.py`
        """
        plot_3D(self, color_scheme, color_mapping, synapses, \
                save_image, show_radius=show_radius)

    def animate(self, color_scheme="default", color_mapping=None,
                synapses=None, save_image="animation", axis="z"):

        """
        Gate way to btviz plot_3D_SWC to create object orientated relationship

        3D matplotlib plot of a neuronal morphology. The SWC has to be
        formatted with a "three point soma".
        Colors can be provided and synapse location marked

        Parameters
        ----------
        color_scheme: string
            "default" or "neuromorpho". "neuronmorpho" is high contrast
        color_mapping: list[float] or list[list[float,float,float]]
            Default is None. If present, this is a list[N] of colors
            where N is the number of compartments, which roughly corresponds to
            the number of lines in the SWC file. If in format of list[float],
            this list is normalized and mapped to the jet color map, if in
            format of list[list[float,float,float,float]], the 4 floats represt
            R,G,B,A respectively and must be between 0-255. When not None, this
            argument overrides the color_scheme argument(Note the difference
            with segments).
        synapses : vector of bools
            Default is None. If present, draw a circle or dot in a distinct
            color at the location of the corresponding compartment. This is
            a 1xN vector.
        save_image: string
            Default is "animation". If present, should be in format
            "file_name", and animation produced will be saved as
            this filename.gif.
        axis: string
            Default is "z". Rotation axis to animate, can be "x","y" or "z"
        """
        animate(self, color_scheme, color_mapping, synapses, save_image, axis)

    def plot_dendrogram(self):
        plot_dendrogram(self)

    def plot_2D(self, color_scheme="default", color_mapping=None,
                synapses=None, save_image=None, depth='y', show_radius=True):

        """
        Gate way to btviz plot_2D_SWC to create object orientated relationship

        2D matplotlib plot of a neuronal moprhology. Projection can be in XY
         and XZ.
        The SWC has to be formatted with a "three point soma".
        Colors can be provided

        Parameters
        -----------
        color_scheme: string
            "default" or "neuromorpho". "neuronmorpho" is high contrast
        color_mapping: list[float] or list[list[float,float,float]]
            Default is None. If present, this is a list[N] of colors
            where N is the number of compartments, which roughly corresponds to
            the number of lines in the SWC file. If in format of list[float],
            this list is normalized and mapped to the jet color map, if in
            format of list[list[float,float,float,float]], the 4 floats represt
            R,G,B,A respectively and must be between 0-255. When not None, this
            argument overrides the color_scheme argument(Note the difference
            with segments).
        synapses : vector of bools
            Default is None. If present, draw a circle or dot in a distinct
            color at the location of the corresponding compartment. This is a
            1xN vector.
        save_image: string
            Default is None. If present, should be in format
            "file_name.extension", and figure produced will be saved as
            this filename.
        depth : string
            Default 'y' means that X represents the superficial to deep axis. \
            Otherwise, use 'z' to conform the mathematical standard of having
            the Z axis.

        Notes
        -----
        If the soma is not located at [0,0,0], the scale bar (`bar_L`) and the
         ticks (`bar`) might not work as expected

        """
        plot_2D(self, color_scheme, color_mapping, synapses, \
                save_image, depth, show_radius=show_radius)

    def get_points_of_interest(self):
        """
        Get lists containting the "points of interest", i.e., soma points,
        bifurcation points and end/terminal points.

        Returns
        -------
        soma_points : list
        bif_points : list
        end_points : list

        """
        soma_points = []
        bif_points = []
        end_points = []

        # updated 2014-01-21 for compatibility with new btstructs
        for node in self._all_nodes:
            if len(node.children) > 1:
                if node.parent is not None:
                    bif_points.append(node)  # the root is not a bifurcation
            if len(node.children) == 0:
                if node.parent.index != 1:  # "3 point soma",
                    # avoid the two side branches
                    end_points.append(node)
            if node.parent is None:
                soma_points = [node]

        return soma_points, bif_points, end_points

    def approx_soma(self):
        """
        *Scalar, global morphometric*

        By NeuroMorpho.org convention: soma surface ~ 4*pi*r^2, \
        where r is the abs(y_value) of point 2 and 3 in the SWC file


        Returns
        -------
        surface : float
             soma surface in micron squared
        """

        r = self.__tree.get_node_with_index(1).content['p3d'].radius
        return 4.0 * np.pi * r * r

    def no_bifurcations(self):
        """
        *Scalar, global morphometric*

        Count the number of bifurcations points in a complete moprhology

        Returns
        -------
        no_bifurcations : int
             number of bifurcation
        """
        return len(self._bif_points)

    def no_terminals(self):
        """
        *Scalar, global morphometric*

        Count the number of temrinal points in a complete moprhology

        Returns
        -------
        no_terminals : int
            number of terminals
        """
        return len(self._end_points)

    def no_stems(self):
        """
        *Scalar, global morphometric*

        Count the number of stems in a complete moprhology (except the three \
        point soma from the Neuromoprho.org standard)

        Returns
        -------
        no_stems : int
            number of stems
        """
        return len(self.__tree.root.children) - 2

    def no_nodes(self):
        """
        *Scalar, global morphometric*

        Count the number of nodes in a complete moprhology

        Returns
        -------
        no_nodes: int
            number of stems
        """
        return self.__tree.get_nodes().__len__()

    def total_length(self):
        """
        *Scalar, global morphometric*

        Calculate the total length of a complete morphology


        Returns
        -------
        total_length : float
            total length in micron
        """
        L = 0
        # updated 2014-01-21 for compatibility with new btstructs
        for Node in self._all_nodes:
            n = Node.content['p3d']
            if Node.index not in (1, 2, 3):
                p = Node.parent.content['p3d']
                d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
                L += d

        return L

    def total_surface(self):
        """
        *Scalar, global morphometric*

        Total neurite surface (at least, surface of all neurites excluding
        the soma. In accordance to the NeuroMorpho / L-Measure standard)

        Returns
        -------
        total_surface : float
            total surface in micron squared

        """
        total_surf = 0
        all_surfs = []
        # updated 2014-01-21 for compatibility with new btstructs
        for Node in self._all_nodes:
            n = Node.content['p3d']
            if Node.index not in (1, 2, 3):
                p = Node.parent.content['p3d']
                H = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
                surf = 2 * np.pi * n.radius * H
                all_surfs.append(surf)
                total_surf = total_surf + surf
        return total_surf, all_surfs

    def total_volume(self):
        """
        *Scalar, global morphometric*

        Total neurite volume (at least, surface of all neurites excluding
        the soma. In accordance to the NeuroMorpho / L-Measure standard)

        Returns
        -------
        total_volume : float
            total volume in micron cubed
        """
        total_vol = 0
        all_vols = []
        # updated 2014-01-21 for compatibility with new btstructs
        for Node in self._all_nodes:
            n = Node.content['p3d']
            if Node.index not in (1, 2, 3):
                p = Node.parent.content['p3d']
                H = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
                vol = np.pi * n.radius * n.radius * H
                all_vols.append(vol)
                total_vol = total_vol + vol
        return total_vol, all_vols

    def total_dimension(self):
        """
        *Scalar, global morphometric* Overall dimension of the morphology

        Returns
        -------
        dx : float
            x-dimension
        dy : float
            y-dimension
        dz : float
            z-dimension
        """
        dx, dy, dz, maxs = self.total_dimensions_verbose()
        return dx, dy, dz

    def total_dimensions_verbose(self):
        """
        *Scalar, global morphometric*

        Overall dimension of the whole moprhology. (No translation of the \
        moprhology according to arbitrary axes.)

        Returns
        -------
        dx : float
            x-dimension
        dy : float
            y-dimension
        dz : float
            z-dimension
        data : list
            minX, maxX, minY, maxY, minZ, maxZ
        """
        # comparisons (preset max and min; minint is -maxint - 1, as mentioned
        # here: https://docs.python.org/2/library/sys.html)
        minX = sys.maxsize
        maxX = -sys.maxsize - 1
        minY = sys.maxsize
        maxY = -sys.maxsize - 1
        minZ = sys.maxsize
        maxZ = -sys.maxsize - 1
        for Node in self._all_nodes:
            n = Node.content['p3d']
            nx = n.xyz[0]
            ny = n.xyz[1]
            nz = n.xyz[2]
            minX = nx if nx < minX else minX
            maxX = nx if nx > maxX else maxX

            minY = ny if ny < minY else minY
            maxY = ny if ny > maxY else maxY

            minZ = nz if nz < minZ else minZ
            maxZ = nz if nz > maxZ else maxZ
        dx = np.sqrt((maxX - minX) * (maxX - minX))
        dy = np.sqrt((maxY - minY) * (maxY - minY))
        dz = np.sqrt((maxZ - minZ) * (maxZ - minZ))
        return dx, dy, dz, [minX, maxX, minY, maxY, minZ, maxZ]

    def global_horton_strahler(self):
        """
        Calculate Horton-Strahler number at the root
        See :func:`local_horton_strahler`

        Returns
        ---------
        Horton-Strahler number at the root
        """
        return self.local_horton_strahler(self.__tree.root)

    def max_EucDistance_from_root(self):

        """
        Returns the Euclidean distance of the node which has the maximum
        Euclidean distance from the root.
        """
        return max(list(map(self.get_Euclidean_length_to_root, self._end_points)))

    def max_pathLength_from_root(self):

        """
        Returns the path length of the node which has the maximum path length
        from the root.
        """
        return max(list(map(self.get_pathlength_to_root, self._end_points)))

    def max_centrifugal_order(self):
        """
        Returns the maximum of the centrifugal orders of all nodes in the tree.
        """
        return max(list(map(self.order_of_node, self._end_points)))

    def avg_bif_angle_local(self):
        """
        Returns the average of the local bifurcation angles of all bifurcation nodes
        in the tree.

        Return
        ------
        (average bifurcation angle local, all bifurcation angles local): (float, list)
        """
        if len(self._bif_points):
            bifAngles = [self.bifurcation_angle_vec(n, where="local") for n in self._bif_points]
            return float(np.mean(bifAngles)), bifAngles
        else:
            return float('nan'), [float("nan")]

    def avg_bif_angle_remote(self):
        """
        Returns the average of the remote bifurcation angles of all bifurcation nodes
        in the tree.

        Return
        ------
        (average bifurcation angle local, all bifurcation angles local): (float, list)
        """
        if len(self._bif_points):
            bifAngles = [self.bifurcation_angle_vec(n, where="remote") for n in self._bif_points]
            return float(np.mean(bifAngles)), bifAngles
        else:
            return float('nan'), [float("nan")]

    def avg_partition_asymmetry(self):
        """
        Returns the average of the partition assymetries of all bifurcation
        nodes in the tree.
        """
        if len(self._bif_points):

            return float(np.mean(list(map(self.partition_asymmetry, self._bif_points))))
        else:
            return float('nan')

    def avg_diameter(self):
        """
        Returns the average of the diameters of all nodes in the tree.
        """
        return float(np.mean(self.get_diameters()))

    def avg_Burke_taper(self):
        """
        Calculate the average Burke taper of all paths in the tree.
        A path is defined as a stretch between
        the soma and a bifurcation point, between bifurcation points,
        or in between of a bifurcation point and a terminal point

        Returns
        -------
        (average_Burke_taper, all_Burke_tapers): (float, list)
            A tuple of the average Burke taper for the tree and a list of Burke
            tapers of all paths of the tree.
        """

        burkeTapers = list(map(self.Burke_taper, self._end_points + self._bif_points))

        return float(np.mean(burkeTapers)), burkeTapers

    def avg_contraction(self):
        """
        Calculate the average contraction of all paths in the tree.
        A path is defined as a stretch between
        the soma and a bifurcation point, between bifurcation points,
        or in between of a bifurcation point and a terminal point

        Returns
        -------
        (average contraction, all contraction values): (float, list)
            A tuple of the average contraction for all paths in the tree and a
            list of Burke tapers of all paths in the tree.
        """

        contractions = list(map(self.contraction, self._end_points + self._bif_points))

        return float(np.nanmean(contractions)), contractions

    def avg_sibling_ratio_local(self):
        """
        Calculate the average of ratio of diameters of sibling branches at all bifurcations.
        Returns
        -------
        (average_sibling_ratio_local, all_sibling_ratio_local): (float, list)
        """

        if len(self._bif_points):
            siblingRatios_local = [self.bifurcation_sibling_ratio(n, where='local') for n in self._bif_points]

            return float(np.mean(siblingRatios_local)), siblingRatios_local
        else:
            return float("nan"), [float("nan")]

    """
    Local measures
    """

    def get_diameters(self):
        """
        *Vector, local morphometric*

        Get the diameters of all points in the morphology
        """
        diams = []
        for node in self._all_nodes:
            if node.index not in (1, 2, 3):
                diams.append(node.content['p3d'].radius * 2.0)
        return diams

    def get_segment_pathlength(self, to_node):
        """
        *Vector, local morphometric*.

        Length of the incoming segment. Between this Node and the soma or
        another branching point. A path is defined as a stretch between
        the soma and a bifurcation point, between bifurcation points,
        or in between of a bifurcation point and a terminal point

        Parameters
        ----------
        to_node : :class:`btmorph.btstructs.SNode`
           Node *to* which the measurement is taken

        Returns
        -------
        length : float
            length of the incoming path in micron
        """
        # updated 2014-01-21 for compatibility with new btstructs
        L = 0
        if self.__tree.is_leaf(to_node):
            path = self.__tree.path_to_root(to_node)
            L = 0
        else:
            path = self.__tree.path_to_root(to_node)[1:]
            p = to_node.parent.content['p3d']
            n = to_node.content['p3d']
            d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
            L = L + d

        for node in path:
            # print 'going along the path'
            n = node.content['p3d']
            if len(node.children) >= 2:  # I arrive at either the soma or a
                # branch point close to the soma
                return L
            else:
                p = node.parent.content['p3d']
                d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
                L = L + d

    def get_pathlength_to_root(self, from_node):
        """
        Length of the path between from_node to the root.
        another branching point

        Parameters
        ----------
        from_node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        length : float
            length of the path between the soma and the provided Node
        """

        L = 0
        if self.__tree.is_leaf(from_node):
            path = self.__tree.path_to_root(from_node)
            L = 0
        else:
            path = self.__tree.path_to_root(from_node)[1:]
            p = from_node.parent.content['p3d']
            n = from_node.content['p3d']
            d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
            L = L + d

        for node in path[:-1]:
            # print 'going along the path'
            n = node.content['p3d']
            p = node.parent.content['p3d']
            d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
            L = L + d
        return L

    def get_segment_Euclidean_length(self, to_node):
        """
        Euclidean length to the incoming segment. Between this Node and the
         soma or another branching point

        Parameters
        ----------
        from_node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        length : float
            Euclidean distance *to* provided Node (from soma or first branch
             point with lower order)
        """
        L = 0
        if self.__tree.is_leaf(to_node):
            path = self.__tree.path_to_root(to_node)
        else:
            #TODO for nodes that are direct children of the root, results in path being a Node (??)
            # leading to returning a value of 0
            path = self.__tree.path_to_root(to_node)[1:]

        n = to_node.content['p3d']
        for Node in path:
            if len(Node.children) >= 2:
                return L
            else:
                p = Node.parent.content['p3d']
                d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
                L = d

    def get_Euclidean_length_to_root(self, from_node):
        """
        Euclidean length between the from_node and the root

        Parameters
        ----------
        from_node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        length : float
            length of the path between the soma and the provided Node

        """
        n = from_node.content['p3d']
        p = self.__tree.root.content['p3d']
        d = np.sqrt(np.sum((n.xyz - p.xyz) ** 2))
        return d

    def max_degree(self):
        # -1: subtract the 2 fake nodes from the 3-point soma position
        return self.degree_of_node(self.tree.root) - 2

    def degree_of_node(self, node):
        """
        Degree of a Node. (The number of leaf Node in the subneuron mounted at
         the provided Node)

        Parameters
        ----------
        node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        degree : float
        """
        return self.tree.degree_of_node(node)

    def order_of_node(self, node):
        """
        Order of a Node. (Going centrifugally away from the soma, the order
         increases with 1 each time a bifurcation point is passed)

        Parameters
        ----------
        node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        order : float
            order of the subneuron rooted at Node
        """
        return self.__tree.order_of_node(node)

    def partition_asymmetry(self, node):
        """
        *Vector, local morphometric*

        Compute the partition asymmetry for a given Node.

        Parameters
        ----------
        node : :class:`btmorph.btstructs.SNode`

        Returns
        -------
        partition_asymmetry : float
            partition asymmetry of the subneuron rooted at Node
            (according to vanpelt and schierwagen 199x)
        """
        if node.children is None or len(node.children) == 1:
            return None
        d1 = self.__tree.degree_of_node(node.children[0])
        d2 = self.__tree.degree_of_node(node.children[1])
        if (d1 == 1 and d2 == 1):
            return 0  # by definition
        else:
            return old_div(np.abs(d1 - d2), (d1 + d2 - 2.0))

    def amp(self, a):
        return np.sqrt(np.sum((a) ** 2))

    def bifurcation_angle_vec(self, node, where='local'):
        """
        *Vector, local morphometric*

        Only to be computed at branch points (_bif_points). Computes the angle
        between the two daughter branches in the plane defined by the
        parent and the two daughters.

        cos alpha = :math:`(a \dot b) / (|a||b|)`

        Parameters
        -----------
        Node : :class:`btmorph.btstructs.Node`
        where : string
            either "local" or "remote". "Local" uses the immediate daughter
            points while "remote" uses the point just before the next
            bifurcation or terminal point.

        Returns
        -------
        angle : float
            Angle in degrees
        """
        child_node1, child_node2 = self._get_child_nodes(node, where=where)
        scaled_1 = child_node1.content['p3d'].xyz - node.content['p3d'].xyz
        scaled_2 = child_node2.content['p3d'].xyz - node.content['p3d'].xyz
        return (old_div(np.arccos(old_div(np.dot(scaled_1, scaled_2),
                                          (self.amp(scaled_1) * self.amp(scaled_2)))),
                        (2 * np.pi / 360)))

    def bifurcation_sibling_ratio(self, node, where='local'):
        """
        *Vector, local morphometric*

        Ratio between the diameters of two siblings.

        Parameters
        ----------
        Node : :class:`btmorph.btstructs.SNode`, bifurcation node at which the ratio is calcuated
        where : string
            Toggle 'local' or 'remote'

        Returns
        -------
        result : float
            Ratio between the diameter of two siblings
        """
        child1, child2 = self._get_child_nodes(node, where=where)
        radius1 = child1.content['p3d'].radius
        radius2 = child2.content['p3d'].radius
        if radius1 > radius2:
            return old_div(radius1, radius2)
        else:
            return old_div(radius2, radius1)

    def _get_child_nodes(self, node, where):
        if where == 'local':
            return node.children[0], node.children[1]
        else:
            grandchildren = []
            for child in node.children:
                t_child = self._find_remote_child(child)
                grandchildren.append(t_child)
        return grandchildren[0], grandchildren[1]

    def _find_remote_child(self, node):
        t_node = node
        while len(t_node.children) < 2:
            if len(t_node.children) == 0:
                # print t_node, '-> found a leaf'
                return t_node
            t_node = t_node.children[0]
        # print t_node,' -> found a bif'
        return t_node

    def bifurcation_ralls_power_fmin(self, node, where='local'):
        """
        *Vector, local morphometric*

        Approximation of Rall's ratio using scipy.optimize.fmin.
        The error function is :math:`F={D_{d1}}^n+{D_{d2}}^n-{D_p}^n`

        Parameters
        ----------
        node : :class:`btmorph.btstructs.SNode`
        where : string
            either "local" or "remote". "Local" uses the immediate daughter
            points while "remote" uses the point just before the next
            bifurcation or terminal point.

        Returns
        -------
        rr : float
            Approximation of Rall's ratio
        """
        p_diam = node.content['p3d'].radius * 2
        child1, child2 = self._get_child_nodes(node, where=where)
        d1_diam = child1.content['p3d'].radius * 2
        d2_diam = child2.content['p3d'].radius * 2
        # print 'pd=%f,d1=%f,d2=%f' % (p_diam,d1_diam,d2_diam)

        if d1_diam >= p_diam or d2_diam >= p_diam:
            return np.nan

        import scipy.optimize
        mismatch = lambda n: np.abs(np.power(d1_diam, n) +
                                    np.power(d2_diam, n) -
                                    np.power(p_diam, n))
        p_lower = 0.0
        p_upper = 5.0  # THE associated mismatch MUST BE NEGATIVE

        best_n = scipy.optimize.fmin(mismatch,
                                     old_div((p_upper - p_lower), 2.0),
                                     disp=False)
        if 0.0 < best_n < 5.0:
            return best_n
        else:
            return np.nan

    def bifurcation_rall_ratio_classic(self, node, where='local'):
        """
        *Vector, local morphometric*

        The ratio :math:`\\frac{ {d_1}^p + {d_2}^p  }{D^p}` computed with
        :math:`p=1.5`

        Parameters
        -----------
        node : :class:`btmorph.btstructs.SNode`
        where : string
            either 'local or 'remote'. 'Local' uses the immediate daughter
            points while "remote" uses the point just before the next
            bifurcation or terminal point.

        Returns
        -------
        rr : float
            Approximation of Rall's ratio

        """
        p_diam = node.content['p3d'].radius * 2
        child1, child2 = self._get_child_nodes(node, where=where)
        d1_diam = child1.content['p3d'].radius * 2
        d2_diam = child2.content['p3d'].radius * 2

        return (old_div((np.power(d1_diam, 1.5) + np.power(d2_diam, 1.5)),
                        np.power(p_diam, 1.5)))

    def bifurcation_ralls_power_brute(self, node, where='local', min_v=0,
                                      max_v=5, steps=1000):
        """
        *Vector, local morphometric*

        Approximation of Rall's ratio.
         :math:`D^p = {d_1}^p + {d_2}^p`, p is approximated by brute-force
         checking the interval [0,5] in 1000 steps (by default, but the exact
         search dimensions can be specified by keyworded arguments.

        Parameters
        -----------
        node : :class:`btmorph.btstructs.SNode`
        where : string
            either 'local or 'remote'. 'Local' uses the immediate daughter
            points while "remote" uses the point just before the next
            bifurcation or terminal point.

        Returns
        -------
        rr : float
            Approximation of Rall's power, p

        """
        p_diam = node.content['p3d'].radius * 2
        child1, child2 = self._get_child_nodes(node, where=where)
        d1_diam = child1.content['p3d'].radius * 2
        d2_diam = child2.content['p3d'].radius * 2
        # print 'pd=%f,d1=%f,d2=%f' % (p_diam,d1_diam,d2_diam)

        if d1_diam >= p_diam or d2_diam >= p_diam:
            return None

        test_v = np.linspace(min_v, max_v, steps)
        min_mismatch = 100000000000.0
        best_n = -1
        for n in test_v:
            mismatch = ((np.power(d1_diam, n) + np.power(d2_diam, n)) -
                        np.power(p_diam, n))
            # print "n=%f -> mismatch: %f" % (n,mismatch)
            if np.abs(mismatch) < min_mismatch:
                best_n = n
                min_mismatch = np.abs(mismatch)
        return best_n

    def pos_angles(self, x):
        return x if x > 0 else 180 + (180 + x)

    def _get_ampl_angle(self, node):
        """
        Compute the angle of this Node on the XY plane and against the origin
        """
        a = np.rad2deg(np.arctan2(node.content['p3d'].y,
                                  node.content['p3d'].x))
        return self.pos_angle(a)

    def local_horton_strahler(self, node):
        """
        We assign Horton-Strahler number to all nodes of a neuron,
         in bottom-up order, as follows:

        If the Node is a leaf (has no children), its Strahler number is one.
        If the Node has one child with Strahler number i, and all other
         children have Strahler numbers less than i, then the Strahler number
         of the Node is i again.
        If the Node has two or more children with Strahler number i, and no
         children with greater number, then the Strahler number of the Node
         is i + 1.
        *If the Node has only one child, the Strahler number of the Node equals
         to the Strahler number of the child
        The Strahler number of a neuron is the number of its root Node.

        See wikipedia for more information: http://en.wikipedia.org/
        wiki/Strahler_number

        Parameters
        ---------
        node : :class:`btmorph.btstructs.SNode`
            Node of interest
        Returns
        ---------
        hs : int
            The Horton-Strahler number (Strahler number) of the Node
        """
        # Empy neuron
        if node is None:
            return -1
        # Leaf => HS=1
        if len(node.children) == 0:
            return 1
        # Not leaf
        childrenHS = list(map(self.local_horton_strahler, node.children))
        return max(childrenHS + [(min(childrenHS) + 1)])

    def Burke_taper(self, node):
        """
        Calculate burke tapers of the path ending at the given node.
        A path is defined as a stretch between
        the soma and a bifurcation point, between bifurcation points,
        or between a bifurcation point and a terminal point

        Burke taper = (d_e - d_s) / l
        where d_e and d_s are the diameters at the end and start, respectively,
        of a path.


        Ref: Burke, R E, W B Marks, and B Ulfhake.
        "A Parsimonious Description of Motoneuron Dendritic Morphology Using
        Computer Simulation."
        The Journal of neuroscience (1992)

        Parameters
        ---------
        node : :class:`btmorph.btstructs.SNode`
            Node of interest

        Returns
        -------
        List of burke tapers, each corresponding to one child
        """
        assert node in self._end_points + self._bif_points, \
            'Burke Taper can only be calculated for the end_point or a ' + \
            'bifurcation.'

        d_e = 2 * node.content['p3d'].radius

        if self.__tree.is_leaf(node):
            path = self.__tree.path_to_root(node)
        else:
            path = self.__tree.path_to_root(node)[1:]

        remote_parent = path[-1]
        for n in path:
            if len(n.children) >= 2:
                remote_parent = n

        d_s = 2 * remote_parent.content['p3d'].radius
        pathLength = self.get_segment_pathlength(node)

        burke_taper = old_div((d_e - d_s), pathLength)

        return burke_taper

    def contraction(self, node):
        """
        Calculate the contraction of the path ending at the node.
        A path is defined as a stretch between
        the soma and a bifurcation point, between bifurcation points,
        or in between of a bifurcation point and a terminal point

        contraction = (Euclidean distance between the ends of the path)/ (path length of the path)


        Parameters
        ---------
        node : :class:`btmorph.btstructs.SNode`
            Node of interest

        Returns
        -------
        List of contraction values
        """
        assert node in self._end_points + self._bif_points, \
            'Contraction can only be calculated for an end point or at a ' + \
            'bifurcation.'

        pathLen = self.get_segment_pathlength(node)
        eucLen = self.get_segment_Euclidean_length(node)

        if eucLen == 0:
            return float("nan")
        else:
            return old_div(eucLen, pathLen)

    def get_boundingbox(self):
        '''
        Get minimum and maximum positions in each axis

        Returns
        --------
        minv : 1D array of 3 int
            minimum values of axis in order x,y,z
        maxv : 1D array of 3 int
            maximum values of axis in order x,y,z
        '''
        minv = [0, 0, 0]
        maxv = [0, 0, 0]
        for node in self._all_nodes:
            xyz = node.content['p3d'].xyz
            for i in (0, 1, 2):
                if xyz[i] < minv[i]:
                    minv[i] = xyz[i]
                if xyz[i] > maxv[i]:
                    maxv[i] = xyz[i]
        return minv, maxv


    def affineTransform(self, affineTransformationMatrix):
        """
        Returns a copy of self that is transformed by affineTransformMatrix
        :param affineTransformationMatrix: numpy.ndarray of shape (4, 4)
        :return: Neuron Morphology
        """

        newNM = NeuronMorphology()
        newNM.tree = self.tree.affineTransformTree(affineTransformationMatrix)
        return newNM


    def getGlobalScalarMeasures(self, funcs=None):
        """
        Collect global scalar measures and return them as a dictionary
        :param funcs: list of strings, containing method names of Class NeuronMorphology to use.
        :return: dict, with function names as dict keys and corresponding measure values as dict values.
        """

        if funcs is None:
            funcs = defaultGlobalScalarFuncs

        swcDataS = {}

        totalDimDict = {'width': 0, 'height': 1, 'depth': 2}

        funcsSet = set(funcs)
        totalDimRequired = set(totalDimDict.keys()).intersection(funcsSet)

        if len(totalDimRequired):
            totalDims = self.total_dimension()
            for tD in totalDimRequired:
                swcDataS[tD] = totalDims[totalDimDict[tD]]

        for func in funcsSet.difference(totalDimRequired):

            method = getattr(self, func)
            ret = method()
            if func in ['total_surface', 'total_volume', 'avg_Burke_taper', 'avg_contraction',
                        'avg_sibling_ratio_local', 'avg_bif_angle_local', 'avg_bif_angle_remote']:
                ret = ret[0]
            swcDataS[func] = ret

        return swcDataS

    def getIntersectionsVsDistance(self, radii, centeredAt=None):
        """
        Calculates and returns the number of intersections of the morphology with concentric spheres of radii in
        input argument radii
        :param radii: iterable of non-negative floats of size at least 2, radii of spheres concerned
        :param centeredAt: iterable of floats of size 3, containing the X, Y and Z coordinates of the center of the
        spheres
        :return: list of same size as radii, of intersections, corresponding to radii in input argument radii
        """

        assert len(radii) >= 2, "Input argument radii must have at least 2 numbers, got {}".format(radii)
        assert all([x >= 0 for x in radii]), "Input argument radii can only consist of non-negative numbers, " \
                                             "got {}".format(radii)
        intersects = [0 for x in radii]
        radii = list(radii)
        radiiSorted = np.sort(radii)

        if centeredAt is None:
            centeredAt = self.get_tree().root.content["p3d"].xyz

        assert len(centeredAt) == 3, "Input argument centeredAt must be a 3 member iterable of numbers, " \
                                     "got {}".format(centeredAt)
        centeredAt = np.asarray(centeredAt)

        def nodeDistance(n):
            nXYZ = np.asarray(n.content["p3d"].xyz)
            return np.linalg.norm(nXYZ - centeredAt)

        def nodeXYZ(n):
            return np.asarray(n.content["p3d"].xyz)

        allNodesExceptRoot = [x for x in self.get_tree().breadth_first_iterator_generator() if x.parent]

        for node in allNodesExceptRoot:

            nodeDist = nodeDistance(node)
            parentDist = nodeDistance(node.parent)

            # in case node is farther than the parent
            if nodeDist > parentDist:
                fartherDist = nodeDist
                fartherXYZ = nodeXYZ(node)
                nearerDist = parentDist
                nearerXYZ = nodeXYZ(node.parent)
            else:
                fartherDist = parentDist
                fartherXYZ = nodeXYZ(node.parent)
                nearerDist = nodeDist
                nearerXYZ = nodeXYZ(node)

            # Finding intersects with spheres can have slight error. To avoid inconsistencies, round distances
            nearerDist = np.round(nearerDist, 3)
            fartherDist = np.round(fartherDist, 3)

            if fartherDist > radiiSorted[0]:
                radiiCrossedMask = np.logical_and(nearerDist < radiiSorted, radiiSorted <= fartherDist)
                radiiCrossed = radiiSorted[radiiCrossedMask]

                if len(radiiCrossed) == 0:
                    radiiCrossed = []
                    largestRLessNearerPoint = radiiSorted[radiiSorted <= nearerDist].max()
                    currentIntersects = getIntersectionXYZs(nearerXYZ, fartherXYZ, centeredAt,
                                                            largestRLessNearerPoint)

                    if len(currentIntersects) == 2:
                        if nearerDist > largestRLessNearerPoint:
                            radiiCrossed.append(largestRLessNearerPoint)
                        if fartherDist > largestRLessNearerPoint:
                            radiiCrossed.append(largestRLessNearerPoint)
                for rad in radiiCrossed:
                    intersects[radii.index(rad)] += 1

        return intersects

    def getLengthVsDistance(self, radii, centeredAt=None):
        """
        Calculates and returns the length of dendrites of the morphology contained within concentric shells defined by
        adjacent values of input argument radii. First shell is the sphere of radius radii[0]
        :param radii: iterable of positive floats of size at least 2, radii "bin edges" of shells
        :param centeredAt: iterable of floats of size 3, containing the X, Y and Z coordinates of the center of the
        spheres
        :return: list of size len(radii), of lengths, corresponding to concentric shells defined by adjacent values
        of  radii.
        """

        assert len(radii) >= 2, "Input argument radii must have at least 2 numbers, got {}".format(radii)
        assert all(x > 0 for x in radii), "Input argument radii can only consist of postive numbers, " \
                                          "got {}".format(radii)
        radii = list(radii)
        lengths = [0 for x in radii]
        assert radii == sorted(radii), "Input argument radii must be sorted"
        radiiSorted = np.array(radii)

        if centeredAt is None:
            centeredAt = self.get_tree().root.content["p3d"].xyz

        assert len(centeredAt) == 3, "Input argument centeredAt must be a 3 member iterable of numbers, " \
                                     "got {}".format(centeredAt)
        centeredAt = np.asarray(centeredAt)

        def nodeDistance(n):
            nXYZ = np.asarray(n.content["p3d"].xyz)
            return np.linalg.norm(nXYZ - centeredAt)

        def nodeXYZ(n):
            return np.asarray(n.content["p3d"].xyz)

        allNodesExceptRoot = [x for x in self.get_tree().breadth_first_iterator_generator() if x.parent]

        for node in allNodesExceptRoot:

            nodeDist = nodeDistance(node)
            parentDist = nodeDistance(node.parent)

            # both points are within first shell
            if nodeDist <= radiiSorted[0] and parentDist <= radiiSorted[0]:
                lengths[0] += np.linalg.norm(nodeXYZ(node) - nodeXYZ(node.parent))

            else:
                if nodeDist < parentDist:
                    nearerPoint = nodeXYZ(node)
                    nearerDist = nodeDist
                    fartherPoint = nodeXYZ(node.parent)
                    fartherDist = parentDist
                else:
                    nearerPoint = nodeXYZ(node.parent)
                    nearerDist = parentDist
                    fartherPoint = nodeXYZ(node)
                    fartherDist = nodeDist

                # # Finding intersects with spheres can have slight error. To avoid inconsistencies, round distances
                # nearerDist = np.round(nearerDist, 3)
                # fartherDist = np.round(fartherDist, 3)

                radiiCrossedMask = np.logical_and(nearerDist <= radiiSorted, radiiSorted <= fartherDist)
                radiiCrossed = radiiSorted[radiiCrossedMask]

                # line connecting the points are within one shell
                if len(radiiCrossed) == 0:

                    largestRLessNearerPoint = radiiSorted[radiiSorted <= nearerDist].max()
                    intersects = getIntersectionXYZs(nearerPoint, fartherPoint, centeredAt, largestRLessNearerPoint)

                    # line joining the points does not intersect any sphere
                    # (len(intersects) == 1 can occur due to rounding error in finding intersects)
                    if len(intersects) < 2:
                        shellIndex = radii.index(largestRLessNearerPoint) + 1
                        lengths[shellIndex] += np.linalg.norm(fartherPoint - nearerPoint)
                    # line joining the points intersects a sphere are two distinct points
                    elif len(intersects) == 2:
                        intersects = np.array(intersects)
                        innerShellIndex = radii.index(largestRLessNearerPoint)
                        outerShellIndex = radii.index(largestRLessNearerPoint) + 1
                        lengths[outerShellIndex] += np.linalg.norm(intersects[0] - nearerPoint)
                        lengths[outerShellIndex] += np.linalg.norm(fartherPoint - intersects[1])
                        lengths[innerShellIndex] += np.linalg.norm(intersects[1] - intersects[0])


                # line connecting the points is contained in at least two shells
                else:
                    tempNearestPoint = nearerPoint
                    for rad in radiiCrossed:
                        shellIndex = radii.index(rad)
                        intersects = getIntersectionXYZs(nearerPoint, fartherPoint, centeredAt, rad)
                        if len(intersects) == 1:
                            intersect = np.array(intersects[0])
                        # len(intersects) == 2 can happen when the line joining the points crosses one sphere
                        # two times and nearerPoint is on the sphere.
                        elif len(intersects) == 2:
                            intersect = np.array(intersects[1])
                        else:
                            # This case must theoretically not happen, but happens due to rounding errors
                            pass
                            # raise(ValueError("Impossible case! There has been a wrong assumption."))


                        lengths[shellIndex] += np.linalg.norm(intersect - tempNearestPoint)
                        tempNearestPoint = intersect
                    if shellIndex + 1 < len(radii):
                        lengths[shellIndex + 1] += np.linalg.norm(fartherPoint - tempNearestPoint)

        return lengths






