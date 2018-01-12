from .btstructs import NeuronMorphology
import pandas as pd


def getGlobalFeautures(swcFiles, funcs=None):
    """
    Calculate Global Features for the SWC Files in swcFiles and return them as a pandas DataFrame
    :param swcFiles: list of strings, list of path strings of SWC Files
    :param funcs: list of strings, containing method names of Class NeuronMorphology to use.
    :return: pandas.DataFrame, with function names as columns and SWC File names as indexes
    """

    statsDF = pd.DataFrame()

    for swcFile in swcFiles:

        nm = NeuronMorphology(input_file=swcFile, correctIfSomaAbsent=True, ignore_type=True)
        nmStatsDict = nm.getGlobalScalarMeasures(funcs)

        for k, v in nmStatsDict.items():

            statsDF.loc[swcFile, k] = v

    return statsDF
