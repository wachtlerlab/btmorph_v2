from __future__ import division
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import object
from .neuronMorph import NeuronMorphology
from six import string_types
from tempfile import mkdtemp
from ..SWCParsing import SWCParsing
import shutil
import pathlib2
from ..auxFuncs import readSWC_numpy


class PopulationMorphology(object):
    '''
    Simple population for use with a simple neuron (:class:`neuron`).

    List of neurons for statistical comparison, no visualisation methods
    '''

    def __init__(self, obj=None,
                 correctIfSomaAbsent=False,
                 ignore_type=False):
        """
        Default constructor.

        Parameters
        -----------
        obj : : str, NeuronMorphology, list[NeuronMorphology], None
            If obj is str it can either be a SWC file or directory containing
            SWC files. If obj is NeuronMorphology then Population will be
            create with NeuronMorphology, if List of NeuronMorphology then
            population will be created with that list. If obj is None, an empty
            PopulationMophology is created [default].
        correctIfSomaAbsent: bool
            if True, then for trees whose roots are not of type 1, the roots are
            manually set to be of type 1 and treated as they have one point soma.
        ignore_type: bool
            if True, the 'type' value in the second column is ignored
        """

        self.neurons = []

        if isinstance(obj, NeuronMorphology):
            self.add_neuron(obj)

        elif isinstance(obj, string_types):
            from os import listdir
            from os.path import isfile, isdir, join

            if isdir(obj):
                files = [f for f in listdir(obj) if (isfile(join(obj, f))
                                                     and
                                                     f.endswith('.swc'))]
                for f in files:
                    nms = self.parseSWCFile2NM(join(obj, f),
                                               correctIfSomaAbsent=correctIfSomaAbsent,
                                               ignore_type=ignore_type)

                    for n in nms:
                        self.add_neuron(n)
            if isfile(obj) and obj.endswith(".swc"):
                nms = self.parseSWCFile2NM(obj,
                                           correctIfSomaAbsent=correctIfSomaAbsent,
                                           ignore_type=ignore_type)

                for n in nms:
                    self.add_neuron(n)

        elif isinstance(obj, list):
            if isinstance(obj[0], NeuronMorphology):
                for n in obj:
                    self.add_neuron(n)

        elif obj is None:
            pass

        else:
            print("Object is not valid type")

    @staticmethod
    def parseSWCFile2NM(swcFile, correctIfSomaAbsent, ignore_type):

        swcP = SWCParsing(swcFile)
        tmpDir = mkdtemp()
        files = swcP.getTreesAsFiles(tmpDir)

        NMs = []
        for f in files:
            n = NeuronMorphology(input_file=f,
                                 correctIfSomaAbsent=correctIfSomaAbsent,
                                 ignore_type=ignore_type)
            NMs.append(n)

        shutil.rmtree(tmpDir)

        return NMs

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def remove_neuron(self, neuron):
        self.neurons.remove(neuron)

    def no_of_neurons(self):
        return len(self.neurons)

    def no_of_bifurcations(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.no_bifurcations())
        return result

    def no_terminals(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.no_terminals())
        return result

    def no_stems(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.no_stems())
        return result

    def total_length(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.total_length())
        return result

    def total_surface(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.total_surface())
        return result

    def total_volume(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.total_volume())
        return result

    def total_dimensions_verbose(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.total_dimensions_verbose())
        return result

    def global_horton_strahler(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.global_horton_strahler())
        return result

    def get_diameters(self):
        result = []
        if self.neurons is not None:
            for n in self.neurons:
                result.append(n.get_diameters())
        return result

    def write_to_SWC_file(self, outFile):

        tmpDir = mkdtemp()
        tmpDirPath = pathlib2.Path(tmpDir)

        with open(outFile, 'w') as outFileObj:
            currentMax = 0
            for nInd, n in enumerate(self.neurons):

                tmpFle = str(tmpDirPath / "{:02d}.swc".format(nInd))

                n.tree.write_SWC_tree_to_file(tmpFle)

                headr, swcData = readSWC_numpy(tmpFle)

                swcData[:, 0] += currentMax
                swcData[1:, 6] += currentMax

                currentMax = swcData[:, 0].max()

                for row in swcData:
                    outFileObj.write('{:.0f} {:.0f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:.0f}\n'.format(*row[:7]))

        shutil.rmtree(tmpDir)

    def affineTransform(self, affineTransformMatrix):
        """
        returns a copy of the Population Morphology with each of its neurons transformed by
        affineTransformMatrix
        :param affineTransformMatrix: numpy.ndarray of shape (4, 4)
        :return: Population Morphology
        """

        newNMs = [x.affineTransform(affineTransformMatrix) for x in self.neurons]
        newPM = PopulationMorphology(newNMs)
        return newPM
