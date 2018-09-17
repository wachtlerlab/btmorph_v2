from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from queue import Queue, LifoQueue
import numpy as np
from numpy import mean, dot, cov, linalg, transpose
from ..SWCParsing import SWCParsing
from ..auxFuncs import writeSWC_numpy, transSWC
from tempfile import NamedTemporaryFile




class Tree(object):
    '''
    Tree for use with a Node (:class:`Node`).

    While the class is designed to contain binary trees (for neuronal
    morphologies)the number of children is not limited. As such,
    this is a generic implementation of a tree structure as a linked list.
    '''

    def __init__(self, input_file=None, axis_config=(0, 1, 2),
                 correctIfSomaAbsent=False, ignore_type=False):

        """
        Default constructor.

        Parameters
        -----------
        input_file : :class:`str`
            File name of neuron to be created
        axis_config: tuple of len 3
            Specifying the column indices at which the x, y and z coordinates
            are to be expected respectively.
        correctIfSomaAbsent: bool
            if True, then for trees whose roots are not of type 1, the roots are
            manually set to be of type 1 and treated as they have one point soma.
        ignore_type: bool
            if True, ignore the 'type' value at column 2
        """

        self.correctIfSomaAbsent = correctIfSomaAbsent
        self.ignore_type = ignore_type

        if input_file is not None:
            self.root = None
            self.read_SWC_tree_from_file(input_file,
                                         correctIfSomaAbsent=correctIfSomaAbsent,
                                         ignore_type=ignore_type)
        if (axis_config[0] is not 0 or axis_config[1]
                is not 1 or axis_config[2] is not 2):  # switch axis
            if axis_config[0] == 0:
                self.switch_axis(1, 2)
            if axis_config[1] == 0:
                self.switch_axis(1, 2)
            elif axis_config[1] == 1:
                self.switch_axis(0, 2)
            elif axis_config[1] == 2:
                self.switch_axis(0, 1)
                self.switch_axis(1, 2)
            if axis_config[2] == 2:
                self.switch_axis(0, 1)

    def switch_axis(self, a, b):
        for n in self.get_nodes():
            tA = n.content['p3d'].xyz[a]
            tB = n.content['p3d'].xyz[b]
            n.content['p3d'].xyz[a] = tB
            n.content['p3d'].xyz[b] = tA

    def set_root(self, node):

        """
        Set the root Node of the tree

        Parameters
        -----------
        Node : :class:`Node`
            to-be-root Node
        """
        if node is not None:
            node.parent = None
        self.__root = node

    def get_root(self):

        """
        Obtain the root Node

        Returns
        -------
        root : :class:`Node`
        """
        return self.__root
    root = property(get_root, set_root)

    def is_root(self, node):

        """
        Check whether a Node is the root Node

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if root

        Returns
        --------
        is_root : boolean
            True is the queried Node is the root, False otherwise
        """
        if node.parent is None:
            return True
        else:
            return False

    def is_leaf(self, node):

        """
        Check whether a Node is a leaf Node, i.e., a Node without children

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if leaf Node

        Returns
        --------
        is_leaf : boolean
            True is the queried Node is a leaf, False otherwise
        """
        if len(node.children) == 0:
            return True
        else:
            return False

    def is_branch(self, node):

        """
        Check whether a Node is a branch Node, i.e., a Node with two children

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if branch Node

        Returns
        --------
        is_leaf : boolean
            True is the queried Node is a branch, False otherwise
        """
        if hasattr(node, 'children'):
            if len(node.children) == 2:
                return True
            else:
                return False
        else:
            return None

    def add_node_with_parent(self, node, parent):

        """
        Add a Node to the tree under a specific parent Node

        Parameters
        -----------
        node : :class:`Node`
            Node to be added
        parent : :class:`Node`
            parent Node of the newly added Node
        """
        node.parent = parent
        if parent is not None:
            parent.add_child(node)

    def remove_node(self, node):

        """
        Remove a Node from the tree

        Parameters
        -----------
        node : :class:`Node`
            Node to be removed
        """
        node.parent.remove_child(node)
        self._deep_remove(node)

    def _deep_remove(self, node):
        children = node.children
        node.make_empty()
        for child in children:
            self._deep_remove(child)

    def get_nodes(self):

        """
        Obtain a list of all nodes in the tree

        Returns
        -------
        all_nodes : list of :class:`Node`
        """
        n = []
        self._gather_nodes(self.root, n)
        return n

    def get_segments_fast(self):

        """
        Obtain a list of all segments in the tree
        fast version (doesn't contain overlapping segments

        Returns
        -------
        all_segments : list of list of :class:`Node` for each segment
        """
        leaves = []
        branches = []
        for n in self.get_nodes():
            if self.is_leaf(n):
                leaves.append(n)
            elif self.is_branch(n):
                branches.append(n)

        all_segments = []

        for l in leaves:
            segment = []
            n = l
            segment.append(n)
            while True:
                n = n.get_parent()
                if self.is_branch(n):
                    segment.append(n)
                    all_segments.append(segment)
                    break
                elif self.is_root(n):
                    segment.append(n)
                    all_segments.append(segment)
                    break
                else:
                    segment.append(n)
        for b in branches:
            segment = []
            n = b
            segment.append(n)
            while True:
                n = n.get_parent()
                if self.is_branch(n):
                    segment.append(n)
                    all_segments.append(segment)
                    break
                elif self.is_root(n):
                    segment.append(n)
                    all_segments.append(segment)
                    break
                else:
                    segment.append(n)
        return all_segments

    def get_segments(self):

        """
        Obtain a list of all segments in the tree
        Contains overlapping segments so joints are modelled correctly

        Returns
        -------
        all_segments : list of list of :class:`Node` for each segment
        """
        leaves = []

        for n in self.get_nodes():
            if self.is_leaf(n):
                leaves.append(n)

        all_segments = []

        for l in leaves:
            segment = []
            n = l
            segment.append(n)
            while True:
                n = n.get_parent()
                if self.is_root(n):
                    segment.append(n)
                    all_segments.append(segment)
                    break
                else:
                    segment.append(n)
        return all_segments

    def get_sub_tree(self, fake_root):

        """
        Obtain the subtree starting from the given Node

        Parameters
        -----------
        fake_root : :class:`Node`
            Node which becomes the new root of the subtree

        Returns
        -------
        sub_tree :  NeuronMorphology
            New tree with the Node from the first argument as root Node
        """
        ret = Tree()
        cp = fake_root.__copy__()
        cp.parent = None
        ret.root = cp
        return ret

    def _gather_nodes(self, node, node_list):

        if node is not None:
            node_list.append(node)
            for child in node.children:
                self._gather_nodes(child, node_list)

    def get_node_with_index(self, index):

        """
        Get a Node with a specific name. The name is always an integer

        Parameters
        ----------
        index : int
            Name of the Node to be found

        Returns
        -------
        Node : :class:`Node`
            Node with the specific index
        """
        return self._find_node(self.root, index)

    def get_node_in_subtree(self, index, fake_root):

        """
        Get a Node with a specific name in a the subtree rooted at fake_root.
        The name is always an integer

        Parameters
        ----------
        index : int
            Name of the Node to be found
        fake_root: :class:`Node`
            Root Node of the subtree in which the Node with a given index
            is searched for

        Returns
        -------
        Node : :class:`Node`
            Node with the specific index
        """
        return self._find_node(fake_root, index)

    def _find_node(self, node, index):

        """
        Sweet breadth-first/stack iteration to replace the recursive call.
        Traverses the tree until it finds the Node you are looking for.

        Parameters
        -----------
        node : :class:`Node`
            Node to be found
         index : int
            Name of the Node to be found

        Returns
        -------
        node : :class:`Node`
            when found and None when not found
        """
        stack = []
        stack.append(node)
        while(len(stack) != 0):
            for child in stack:
                if child.index == index:
                    return child
                else:
                    stack.remove(child)
                    for cchild in child.children:
                        stack.append(cchild)
        return None  # Not found!

    def degree_of_node(self, node):

        """
        Get the degree of a given Node. The degree is defined as the number of
        leaf nodes in the subtree rooted at this Node.

        Parameters
        ----------
        node : :class:`Node`
            Node of which the degree is to be computed.

        Returns
        -------
        degree : int
        """
        sub_tree = self.get_sub_tree(node)
        st_nodes = sub_tree.get_nodes()
        leafs = 0
        for n in st_nodes:
            if sub_tree.is_leaf(n):
                leafs = leafs + 1
        return leafs

    def order_of_node(self, node):

        """
        Get the order of a given Node. The order or centrifugal order is
        defined as 0 for the root and increased with any bifurcation.
        Hence, a Node with 2 branch points on the shortest path between
        that Node and the root has order 2.

        Parameters
        ----------
        node : :class:`Node`
            Node of which the order is to be computed.

        Returns
        -------
        order : int
        """
        ptr = self.path_to_root(node)
        order = 0
        for n in ptr:
            if len(n.children) > 1:
                order = order+1
        # order is on [0,max_order] thus subtract 1 from this calculation
        return order - 1

    def path_to_root(self, node):

        """
        Find and return the path between a Node and the root.

        Parameters
        ----------
        node : :class:`Node`
            Node at which the path starts

        Returns
        -------
        path : list of :class:`Node`
            list of :class:`Node` with the provided Node and the root as first
            and last entry, respectively.
        """
        n = []
        self._go_up_from(node, n)
        return n

    def _go_up_from(self, node, n):

        n.append(node)
        if node.parent is not None:
            self._go_up_from(node.parent, n)

    def path_between_nodes(self, from_node, to_node):

        """
        Find the path between two nodes. The from_node needs to be of higher \
        order than the to_node. In case there is no path between the nodes, \
        the path from the from_node to the soma is given.

        Parameters
        -----------
        from_node : :class:`Node`
        to_node : :class:`Node`
        """
        n = []
        self._go_up_from_until(from_node, to_node, n)
        return n

    def _go_up_from_until(self, from_node, to_node, n):

        n.append(from_node)
        if from_node == to_node:
            return
        if from_node.parent is not None:
            self._go_up_from_until(from_node.parent, to_node, n)

    def breadth_first_iterator_generator(self):
        """
        Generator function to produce an iterator that traverses breadth first through nodes of the tree.
        Ex. [x for x in tree.breadth_first_iterator_generator] produces a list of nodes in breadth-first
        traversal order
        """

        nodeQ = Queue()

        nodeQ.put(self.root)

        while not nodeQ.empty():

            node = nodeQ.get()
            children = dict((child.index, child) for child in node.children)
            childrenIndsSorted = sorted(children.keys())
            for childInd in childrenIndsSorted:
                nodeQ.put(children[childInd])
            yield node

    def depth_first_iterator_generator(self):
        """
        Generator function to produce an iterator that traverses depth first through nodes of the tree.
        Ex. [x for x in tree.breadth_first_iterator_generator()] produces a list of nodes in breadth-first
        traversal order
        """

        nodeQ = LifoQueue()

        nodeQ.put(self.root)

        while not nodeQ.empty():

            node = nodeQ.get()
            children = dict((child.index, child) for child in node.children)
            childrenIndsSorted = sorted(children.keys())
            for childInd in childrenIndsSorted:
                nodeQ.put(children[childInd])
            yield node

    def read_SWC_tree_from_file(self, input_file, types=list(range(1, 10)),
                                correctIfSomaAbsent=False, ignore_type=False):

        """
        Non-specific for a "tree data structure"
        Read and load a morphology from an SWC file and parse it into
        an NeuronMorphology object.

        On the NeuroMorpho.org website, 5 types of somadescriptions are
        considered (http://neuromorpho.org/neuroMorpho/SomaFormat.html).
        The "3-point soma" is the standard and most files are converted
        to this format during a curation step. btmorph follows this default
        specificationand the *internal structure of btmorph implements
        the 3-point soma*.

        However, two other options to describe the soma
        are still allowed and available, namely:
        - soma absent: not implemented
        - multiple cylinder: reduces it to a three point soma with the same surface

        Parameters
        -----------
        input_file : :class:`str`
            File name of neuron to be created
        types: iterable of integers
            Specifies the expected values for column 2 of an SWC file
        correctIfSomaAbsent: bool
            if True, then for trees whose roots are not of type 1, the roots are
            manually set to be of type 1 and treated as they have one point soma.
        ignore_type: bool
            if True, the 'type' value of column 2 are ignored


        """


        swc_parsing = SWCParsing(input_file)
        nTrees = swc_parsing.numberOfTrees()

        if nTrees > 1:

            raise ValueError("Given SWC File {} has more than one trees".format(input_file))

        else:

            swcDatasetsTypes = swc_parsing.getSWCDatasetsTypes(correctIfSomaAbsent)

            self.soma_type = list(swcDatasetsTypes.keys())[0]
            swcData = list(swcDatasetsTypes.values())[0]

            all_nodes = dict()
            for line in swcData:

                index = int(line[0])
                swc_type = int(line[1])
                x = float(line[2])
                y = float(line[3])
                z = float(line[4])
                radius = float(line[5])
                parent_index = int(line[6])

                if swc_type in types or ignore_type:
                    tP3D = P3D(np.array([x, y, z]), radius, swc_type)
                    t_node = Node(index)
                    t_node.content = {'p3d': tP3D}
                    all_nodes[index] = (swc_type, t_node, parent_index)
                    if parent_index < 0:
                        if self.root is None:
                            self.root = t_node
                        else:
                            raise ValueError("File {} has two roots!".format(input_file))
                else:
                    # print type,index
                    pass

            # print "len(all_nodes): ", len(all_nodes)

            # IF 1-point soma representation
            if self.soma_type == 0:
                for index, (swc_type, node, parent_index) in all_nodes.items():
                    if parent_index < 0:
                        # Root has already been set above

                        """add 2 extra point because the internal representation
                        relies on the 3-point soma position.
                        Their indices will be 1 and 2 greater, respectively,
                        than the maximum of all indices"""
                        sp = node.content['p3d']
                        """
                         1 1 xs ys zs rs -1
                         2 1 xs (ys-rs) zs rs 1
                         3 1 xs (ys+rs) zs rs 1
                        """
                        pos1 = P3D([sp.xyz[0], sp.xyz[1]-sp.radius,
                                    sp.xyz[2]], sp.radius, 1)
                        pos2 = P3D([sp.xyz[0], sp.xyz[1]+sp.radius,
                                    sp.xyz[2]], sp.radius, 1)
                        maxIndex = max(all_nodes.keys())
                        sub1 = Node(maxIndex + 1)
                        sub1.content = {'p3d': pos1}
                        sub2 = Node(maxIndex + 2)
                        sub2.content = {'p3d': pos2}
                        self.add_node_with_parent(sub1, self.root)
                        self.add_node_with_parent(sub2, self.root)
                    else:
                        parent_node = all_nodes[parent_index][1]
                        self.add_node_with_parent(node, parent_node)

            # IF 3-point soma representation
            elif self.soma_type == 1:
                for index, (swc_type, node, parent_index) in all_nodes.items():
                    if parent_index < 0:
                        # Root has already been set above
                        pass
                    else:
                        parent_node = all_nodes[parent_index][1]
                        self.add_node_with_parent(node, parent_node)
            # IF multiple cylinder soma representation
            elif self.soma_type == 2:
                # Root has already been set above

                # get all some info
                soma_cylinders = []
                connected_to_root = []
                for index, (swc_type, node, parent_index) in all_nodes.items():
                    if swc_type == 1 and parent_index > 0:
                        soma_cylinders.append((node, parent_index))
                        connected_to_root.append(index)

                # make soma
                s_node_1, s_node_2 = self._make_soma_from_cylinders(soma_cylinders,
                                                                    all_nodes)

                # add soma
                self.root.content["p3d"].radius = s_node_1.content["p3d"].radius
                self.add_node_with_parent(s_node_1, self.root)
                self.add_node_with_parent(s_node_2, self.root)

                # add the other points
                for index, (swc_type, node, parent_index) in all_nodes.items():
                    if swc_type == 1:
                        pass
                    else:
                        parent_node = all_nodes[parent_index][1]
                        if parent_node.index in connected_to_root:
                            self.add_node_with_parent(node, self.root)
                        else:
                            self.add_node_with_parent(node, parent_node)

            else:
                raise NotImplementedError("No Soma Found for {}".format(input_file))

            return self

    def write_SWC_tree_to_file(self, output_file):

        """
        Non-specific for a tree.

        Used to write an SWC file from a morphology stored in this
        :class:`NeuronMorphology`. Output uses the 3-point soma standard.

         Parameters
        -----------
        output_file : :class:`str`
            File name to write SWC to

        """
        swcData = []

        for node in self.breadth_first_iterator_generator():
            index = node.index
            p3d = node.content["p3d"]
            nodeType = p3d.segtype
            x, y, z = p3d.xyz
            radius = p3d.radius
            parent = node.parent
            if parent is None:
                parentIndex = -1
            else:
                parentIndex = parent.index

            swcData.append([index, nodeType, x, y, z, radius, parentIndex])

        writeSWC_numpy(output_file, swcData)

    def _make_soma_from_cylinders(self, soma_cylinders, all_nodes):

        """Now construct 3-point soma
        step 1: calculate surface of all cylinders
        step 2: make 3-point representation with the same surface"""

        total_surf = 0
        for (node, parent_index) in soma_cylinders:
            n = node.content["p3d"]
            p = all_nodes[parent_index][1].content["p3d"]
            H = np.sqrt(np.sum((n.xyz-p.xyz)**2))
            surf = 2*np.pi*p.radius*H
            # print "(node %i) surf as cylinder:  %f (R=%f, H=%f), P=%s" %
            # (node.index,surf,n.radius,H,p)
            total_surf = total_surf+surf
        print("found 'multiple cylinder soma' w/ total soma surface={}".format(total_surf))

        # define appropriate radius
        radius = np.sqrt(old_div(total_surf, (4 * np.pi)))
        # print "found radius: ", radius

        s_node_1 = Node(2)
        r = self.root.content["p3d"]
        rp = r.xyz
        s_p_1 = P3D(np.array([rp[0], rp[1]-radius, rp[2]]), radius, 1)
        s_node_1.content = {'p3d': s_p_1}
        s_node_2 = Node(3)
        s_p_2 = P3D(np.array([rp[0], rp[1]+radius, rp[2]]), radius, 1)
        s_node_2.content = {'p3d': s_p_2}

        return s_node_1, s_node_2

    @staticmethod
    def determine_soma_type(file_n):
        """
        Costly method to determine the soma type used in the SWC file.
        This method searches the whole file for soma entries.

        Parameters
        ----------
        file_n : string
            Name of the file containing the SWC description

        Returns
        -------
        soma_type : int
            Integer indicating one of the su[pported SWC soma formats.
            0: 1-point soma
            1: Default three-point soma
            2: multiple cylinder description,
            3: otherwise [not suported in btmorph]
        """
        file = open(file_n, "r")
        somas = 0
        for line in file:
            if not line.startswith('#') :
                split = line.split()
                index = int(split[0].rstrip())
                s_type = int(split[1].rstrip())
                if s_type == 1 :
                    somas = somas +1
        file.close()
        if somas == 3:
            return 1
        elif somas == 1:
            return 0
        elif somas > 3:
            return 2
        else:
            return 3

    def _pca(self, A):
        """ performs principal components analysis
         (PCA) on the n-by-p data matrix A
         Rows of A correspond to observations, columns to variables.

         Returns :
          coeff :
        is a p-by-p matrix, each column containing coefficients
        for one principal component.
          score :
        the principal component scores; that is, the representation
        of A in the principal component space. Rows of SCORE
        correspond to observations, columns to components.
          latent :
        a vector containing the eigenvalues
        of the covariance matrix of A.
        source: http://glowingpython.blogspot.jp/2011/07/
        principal-component-analysis-with-numpy.html
        """
        # computing eigenvalues and eigenvectors of covariance matrix
        M = (A-mean(A.T, axis=1)).T  # subtract the mean (along columns)
        [latent, coeff] = linalg.eig(cov(M))  # attention:not always sorted
        score = dot(coeff.T, M)  # projection of the data in the new space
        return coeff, score, latent

    def pca_project_tree(self, threeD=True):
        """
        Returns a tree which is a projection of the original tree on the plane
         of most variance

        Parameters
        ----------
        tree : :class:`btmorph.btstructs.STree2`
        A tree

        Returns
        --------
        tree : :class:`btmorph.btstructs.STree2`
            New flattened tree
        """
        nodes = self.get_nodes()
        N = len(nodes)
        coords = [n.content['p3d'].xyz for n in nodes]
        points = transpose(coords)
        _, score, _ = self._pca(points.T)
        if threeD is False:
            score[2, :] = [0]*N
        newp = transpose(score)
        # Move soma to origin
        translate = score[:, 0]
        for i in range(0, N):
            nodes[i].content['p3d'].xyz = newp[i] - translate

        import time
        fmt = '%Y_%b_%d_%H_%M_%S'
        now = time.strftime(fmt)
        self.write_SWC_tree_to_file('tmpTree_3d_' + now + '.swc')
        self = self.read_SWC_tree_from_file('tmpTree_3d_' + now + '.swc')
        import os
        os.remove('tmpTree_3d_' + now + '.swc')
        return self

    def affineTransformTree(self, affineTransformMatrix):
        """
        Returns a copy with affineTransformMatrix applied to each node of the tree.
        :param affineTransformMatrix: np.ndarray of shape (4, 4)
        :return: new transformed Neuron Morphology
        """

        tempSWC = NamedTemporaryFile(mode="w")
        self.write_SWC_tree_to_file(tempSWC.name)

        transTempSWC = NamedTemporaryFile(mode="w")

        transSWC(tempSWC.name, affineTransformMatrix[:3, :3],
                 affineTransformMatrix[:3, 3], transTempSWC.name)

        return Tree(transTempSWC.name,
                    ignore_type=self.ignore_type)

    def __iter__(self):

        nodes = []
        self._gather_nodes(self.root, nodes)
        for n in nodes:
            yield n

    def __getitem__(self, index):

        return self._find_node(self.root, index)

    def __str__(self):

        return "Tree ("+str(len(self.get_nodes()))+" nodes)"


class Node(object):

    """
    Simple Node for use with a simple Neuron (NeuronMorphology)

    By design, the "content" should be a dictionary. (2013-03-08)
    """

    def __init__(self, index):
        """
        Constructor.

        Parameters
        -----------
        index : int
           Index, unique name of the :class:`Node`
        """
        self.parent = None
        self.index = index
        self.children = []
        self.content = {}

    def get_parent(self):

        """
        Return the parent Node of this one.

        Returns
        -------
        parent : :class:`Node`
           In case of the root, None is returned.Otherwise a :class:`Node` is
            returned
        """
        return self.__parent

    def set_parent(self, parent):

        """
        Set the parent Node of a given other Node

        Parameters
        ----------
        Node : :class:`Node`
        """
        self.__parent = parent

    parent = property(get_parent, set_parent)

    def get_index(self):

        """
        Return the index of this Node

        Returns
        -------
        index : int
        """
        return self.__index

    def set_index(self, index):

        """
        Set the unique name of a Node

        Parameters
        ----------

        index : int
        """
        self.__index = index

    index = property(get_index, set_index)

    def get_children(self):

        """
        Return the children nodes of this one (if any)

        Returns
        -------
        children : list :class:`Node`
           In case of a leaf an empty list is returned
        """
        return self.__children

    def set_children(self, children):

        """
        Set the children nodes of this one

        Parameters
        ----------

        children: list :class:`Node`
        """
        self.__children = children

    children = property(get_children, set_children)

    def get_content(self):

        """
        Return the content dict of a :class:`Node`

        Returns
        -------
        parent : :class:`Node`
           In case of the root, None is returned.Otherwise a :class:`Node` is
           returned
        """
        return self.__content

    def set_content(self, content):

        """
        Set the content of a Node. The content must be a dict

        Parameters
        ----------
        content : dict
            dict with content. For use in btmorph at least a 'p3d' entry should
             be present
        """
        if isinstance(content, dict):
            self.__content = content
        else:
            raise Exception("Node.set_content must receive a dict")

    content = property(get_content, set_content)

    def add_child(self, child_node):

        """
        add a child to the children list of a given Node

        Parameters
        -----------
        Node :  :class:`Node`
        """
        self.children.append(child_node)

    def make_empty(self):
        """
        Clear the Node. Unclear why I ever implemented this. Probably to cover
         up some failed garbage collection
        """
        self.parent = None
        self.content = {}
        self.children = []

    def remove_child(self, child):
        """
        Remove a child Node from the list of children of a specific Node

        Parameters
        -----------
        Node :  :class:`Node`
            If the child doesn't exist, you get into problems.
        """
        self.children.remove(child)

    def __str__(self):

        return 'Node (ID: '+str(self.index)+')'

    def __lt__(self, other):

        if self.index < other.index:
            return True

    def __le__(self, other):

        if self.index <= other.index:
            return True

    def __gt__(self, other):

        if self.index > other.index:
            return True

    def __ge__(self, other):

        if self.index >= other.index:
            return True

    def __copy__(self):  # customization of copy.copy

        ret = Node(self.index)
        for child in self.children:
            ret.add_child(child)
        ret.content = self.content
        ret.parent = self.parent
        return ret


class P3D(object):

    """
    Basic container to represent and store 3D information
    """

    def __init__(self, xyz, radius, segtype=7):
        """ Constructor.

        Parameters
        -----------

        xyz : numpy.array
            3D location
        radius : float
        segtype : int
            Type asscoiated with the segment according to SWC standards
        """
        self.xyz = xyz
        self.radius = radius
        self.segtype = segtype

    def __str__(self):

        return "P3D [%.2f %.2f %.2f], R=%.2f" % (self.xyz[0], self.xyz[1],
                                                 self.xyz[2], self.radius)
