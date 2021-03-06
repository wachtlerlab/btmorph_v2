import numpy as np
from collections import Counter


# **********************************************************************************************************************

def readSWC_numpy(swcFile):
    '''
    Read the return the header and matrix data in a swcFile
    :param swcFile: filename
    :return: header (string), matrix data (ndarray)
    '''

    headr = ''
    try:

        with open(swcFile, 'r') as fle:
            lne = fle.readline()
            while lne.startswith("#"):
                headr = headr + lne[1:]
                lne = fle.readline()

        headr = headr.rstrip('\n')
    except UnicodeDecodeError as ude:
        print("Warning: The header of {} has non UTF-8 chars. This head is not read".format(swcFile))

    # np.loadtxt can lead to errors if the file contains non utf-8 chars
    # https://stackoverflow.com/questions/22936790/numpy-loadtxt-unicode-and-python-2-or-3


    swcData = np.genfromtxt(swcFile)

    if len(swcData.shape) == 1:
        swcData = swcData.reshape((1, swcData.shape[0]))

    if swcData.shape[1] < 7:
        raise ValueError("Improper data in SWC file {}".format(swcFile))

    return headr, swcData

# **********************************************************************************************************************


def writeSWC_numpy(fName, swcData, headr=''):
    '''
    Write the SWC data in swcData to the file fName with the header headr
    :param fName: string
    :param swcData: 2D numpy.ndarray with 7 or 8 columns
    :param headr: string
    :return:
    '''

    swcData = np.array(swcData)
    assert swcData.shape[1] in [7, 8], 'Width given SWC Matrix data is incompatible.'


    formatStr = '%d %d %0.6f %0.6f %0.6f %0.6f %d'

    if swcData.shape[1] == 8:
         formatStr += ' %0.6f'

    np.savetxt(fName, swcData, formatStr, header=headr, comments='#')

# **********************************************************************************************************************

def getDuplicates(mylist):

    """
    Return a list of duplicate values in mylist
    :param mylist: list
    :return:
    """

    return [k for k, v in list(Counter(mylist).items()) if v > 1]

# **********************************************************************************************************************

def transSWC(fName, A, b, destFle):
    '''
    Generate an SWC file at destFle with each point `x' of the morphology in fName transformed Affinely as Ax+b
    :param fName: string
    :param A: 2D numpy.ndarray of shape (3, 3)
    :param b: 3 member iterable
    :param destFle: string
    :return:
    '''

    headr = ''
    with open(fName, 'r') as fle:
        lne = fle.readline()
        while lne[0] == '#':
            headr = headr + lne[1:]
            lne = fle.readline()

    data = np.loadtxt(fName)
    data[:, 2:5] = np.dot(A, data[:, 2:5].T).T + np.array(b)

    if data.shape[1] == 7:
        formatStr = '%d %d %0.3f %0.3f %0.3f %0.3f %d'
    elif data.shape[1] == 8:
        formatStr = '%d %d %0.3f %0.3f %0.3f %0.3f %d %d'
    else:
        raise TypeError('Data in the input file is of unknown format.')

    np.savetxt(destFle, data, header=headr, fmt=formatStr)

#***********************************************************************************************************************


def transSWC_rotAboutPoint(fName, A, b, destFle, point):
    '''
    Generate an SWC file at destFle with each point `x' of the morphology in fName transformed Affinely as A(x-mu)+b
    where mu is a specified point.
    Essentially, the morphology is centered at a specified point before being Affinely transformed.
    :param fName: string
    :param A: 2D numpy.ndarray of shape (3, 3)
    :param b: 3 member iterable
    :param destFle: string
    :param point: 3 member iterable
    :return:
    '''

    headr = ''
    with open(fName, 'r') as fle:
        lne = fle.readline()
        while lne[0] == '#':
            headr = headr + lne[1:]
            lne = fle.readline()

    data = np.loadtxt(fName)
    pts = data[:, 2:5]
    rotAbout = np.array(point)
    ptsCentered = pts - rotAbout
    data[:, 2:5] = np.dot(A, ptsCentered.T).T + np.array(b) + rotAbout

    if data.shape[1] == 7:
        formatStr = '%d %d %0.3f %0.3f %0.3f %0.3f %d'
    elif data.shape[1] == 8:
        formatStr = '%d %d %0.3f %0.3f %0.3f %0.3f %d %d'
    else:
        raise TypeError('Data in the input file is of unknown format.')

    np.savetxt(destFle, data, header=headr, fmt=formatStr)
#***********************************************************************************************************************


def getIntersectionXYZs(p1, p2, centeredAt, radius):
    """
    Calculates and returns the points of intersection between the line joining the points p1 and p2
    and the circle centered at centeredAt with radius radius. The points are ordered as they would be encountered when
    moving from p1 to c2
    :param p1: 3 member float iterable
    :param p2: 3 member float iterable
    :param centeredAt: 3 member float iterable
    :param radius: float
    :return: iterable of intersections, each interesection being a 3 member float iterable
    """

    # Solving |x + alpha *y| = radius where
    # x is the vector p1 - centeredAt,
    # y is the vector c2 - centeredAt,
    # alpha is a float in [0, 1]

    p1 = np.array(p1)
    p2 = np.array(p2)
    centeredAt = np.array(centeredAt)

    x = p1 - centeredAt
    y = p2 - centeredAt

    modx = np.linalg.norm(x)

    yMx = y - x
    mod_yMx = np.linalg.norm(yMx)

    xDotyMx = np.dot(x, yMx)

    # the above problem reduces to solving A * alpha^2 + B * alpha + C = 0 where

    A = mod_yMx ** 2
    B = 2 * xDotyMx
    C = modx ** 2 - radius ** 2

    roots = np.roots([A, B, C])
    roots = np.round(roots, 3)
    if all(np.isreal(roots)):
        roots = np.sort(roots)

    intersections = [(x + alpha * (y - x)).tolist() for alpha in roots if np.isreal(alpha) and 1 >= alpha >= 0]

    return intersections

#***********************************************************************************************************************


