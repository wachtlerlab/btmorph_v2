from btmorph2.auxFuncs import getIntersectionXYZs
import numpy as np


def getPointAtDistance_test():
    """
    Testing getIntersectionXYZs function
    """

    # case1, two equal real intersections
    parentXYZ = [20, 0, 0]
    childXYZ = [20, -10, 0]
    centeredAt = [0, 0, 0]
    radius = 20
    expectedIntersects = [[20, 0, 0], [20, 0, 0]]
    intersects = getIntersectionXYZs(parentXYZ, childXYZ, centeredAt, radius)
    assert np.shape(expectedIntersects) == np.shape(intersects) and np.allclose(expectedIntersects, intersects,
                                                                                atol=1e-2)

    # case2, one real intersections
    parentXYZ = [10, 0, 0]
    childXYZ = [20, -10, 0]
    centeredAt = [0, 0, 0]
    radius = 15
    expectedIntersects = [[14.35, -4.35, 0.0]]
    intersects = getIntersectionXYZs(parentXYZ, childXYZ, centeredAt, radius)
    assert np.shape(expectedIntersects) == np.shape(intersects) and np.allclose(expectedIntersects, intersects,
                                                                                atol=1e-2)

    # case 3, one real intersections
    parentXYZ = [20, 0, 0]
    childXYZ = [30, 10, 0]
    centeredAt = [0, 0, 0]
    radius = 25
    expectedIntersects = [[24.58, 4.58, 0.0]]
    intersects = getIntersectionXYZs(parentXYZ, childXYZ, centeredAt, radius)
    assert np.shape(expectedIntersects) == np.shape(intersects) and np.allclose(expectedIntersects, intersects,
                                                                                atol=1e-2)

    # case 4, no intersections
    parentXYZ = [50, -10, 0]
    childXYZ = [50, -20, 0]
    centeredAt = [0, 0, 0]
    radius = 50
    expectedIntersects = []
    intersects = getIntersectionXYZs(parentXYZ, childXYZ, centeredAt, radius)
    assert np.shape(expectedIntersects) == np.shape(intersects) and np.allclose(expectedIntersects, intersects,
                                                                                atol=1e-2)

    # case 5, two unequal real intersections
    parentXYZ = [2.5, 7.5, 0]
    childXYZ = [2.5, -7.5, 0]
    centeredAt = [0, 0, 0]
    radius = 5
    expectedIntersects = [[2.5, 4.33, 0.0], [2.5, -4.33, 0.0]]
    intersects = getIntersectionXYZs(parentXYZ, childXYZ, centeredAt, radius)
    assert np.shape(expectedIntersects) == np.shape(intersects) and np.allclose(expectedIntersects, intersects,
                                                                                atol=1e-2)

if __name__ == "__main__":
    intersects = getIntersectionXYZs(p1=[20, 30, 0], p2=[30, 40, 0], radius=50, centeredAt=[0, 0, 0])