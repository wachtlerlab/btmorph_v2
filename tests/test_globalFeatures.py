from btmorph2.globalFeatures import getGlobalFeautures
import pandas as pd


def test_getGlobalFeatures():
    """
    Testing getGlobalFeatures function
    """

    swcFiles = ["tests/v_e_moto1.CNG.swc",
                "tests/v_e_purk2.CNG.swc"]

    resultsDF = getGlobalFeautures(swcFiles)
    resultsDF.sort_index(axis=1, inplace=True)

    expectedResultsXL = "tests/expectedGlobalFeatues.xlsx"

    expectedResultsDF = pd.read_excel(expectedResultsXL, index_col=0, convert_float=False)
    expectedResultsDF.sort_index(axis=1, inplace=True)
    pd.util.testing.assert_frame_equal(resultsDF, expectedResultsDF)

