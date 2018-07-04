"""Image segmentation module
This module provides utilities to extract nuclei and membranes
contours from microscopy images
"""
import numpy as np
from sympy import Line, oo
import pandas as pd
from tyssue.generation import generate_ring
import cv2 as cv


def generate_ring_from_image(brightfield_path, dapi_path,
                             threshold=28, blur=9, rol_window=20):
    """Create an organo mesh of class AnnularSheet from a brightfield image
    and a CellProfiler DAPI analysis csv file

    Parameters
    ----------
    brightfield_path : string
      path to the brightfield image
    dapi_path : string
      path to the CellProfiler output csv file
    threshold: int >=0
      threshold to apply to the brightfield image
    blur: int >=0
      gaussian blur to apply to the brightfield image
    Return
    ----------
    organo : object of class AnnularSheet
      the organo mesh extracted from the data
#    Nf : int >0
#      number of cells in the organo
#    inners : np.array of shape (Nf, 2)
#      coordinates of the vertices of the mesh on the inner ring
#    outers : np.array of shape (Nf, 2)
#      coordinates of the vertices of the mesh on the outer ring
#    centers : np.array of shape (Nf, 2)
#      coordinates of the nuclei centers clockwise
    """

    membrane_dic = extract_membranes(brightfield_path, threshold, blur)
    clockwise_centers = extract_nuclei(dapi_path, membrane_dic['center_inside'],
                                       membrane_dic['raw_inside'],
                                       membrane_dic['img_shape'])
    #rolling mean
    inners = pd.rolling_mean(membrane_dic['inside'], rol_window, min_periods=1)
    outers = pd.rolling_mean(membrane_dic['outside'], rol_window, min_periods=1)

    #defining the organoid using the data we saved above
    Nf = len(clockwise_centers)

    org_center = membrane_dic['center_inside']-\
                 np.full(2, membrane_dic['img_shape'][0]/2.0)

    #compute the vertices of the mesh
    inner_vs, outer_vs = get_bissecting_vertices(clockwise_centers, inners,
                                                 outers, org_center)

    #initialising the mesh
    organo = generate_ring(Nf, membrane_dic['rIn'], membrane_dic['rOut'])

    # adjustement
    organo.vert_df.loc[organo.apical_verts,
                       organo.coords] = (inner_vs[::-1]-np.full(
                           inner_vs.shape, org_center))*0.323
    organo.vert_df.loc[organo.basal_verts,
                       organo.coords] = (outer_vs[::-1]-np.full(
                           outer_vs.shape, org_center))*0.323
    return organo, inners, outers


def extract_membranes(brightfield_path, threshold=28, blur=9):
    """
    Parameters
    ----------
    brightfield_path : string
      path to the brightfield image
    threshold : int >=0
    value of the threshold to apply to the brightfield image
    blur : int >=0
    value of the gaussian blur to apply to the brightfield image

    Return
    ----------
    inside : np.ndarray
      the extracted contours for the inner rings
    outside : np.ndarray
      the extracted contours for the outer rings

    """
    img = cv.imread(brightfield_path, cv.IMREAD_GRAYSCALE).copy()
    #28
    ret, img = cv.threshold(img, threshold, 255, 0)
    #9
    img = cv.GaussianBlur(img, (blur, blur), 0)

    img, contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                               cv.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)

    contours_length = np.array([c.size for c in contours])
    membrane_ind = np.argsort(contours_length)[-2:]

    retained_contours = contours[membrane_ind]

    circles = np.array((cv.minEnclosingCircle(retained_contours[0]),
                        cv.minEnclosingCircle(retained_contours[1])))

    centers = circles[:, 0]
    radii = circles[:, 1]

    inside, outside, res_dic = _recognize_in_from_out(retained_contours,
                                                      centers, radii)

    inside = np.concatenate((inside, _fill_gaps(inside, 2)))
    outside =  np.concatenate((outside, _fill_gaps(outside, 2)))

    res_dic['raw_inside'] = inside
    res_dic['raw_outside'] = outside

    res_dic['inside'] = (inside - np.ones(inside.shape)*(img.shape[0]/2.0,
                                                         img.shape[1]/2.0))
    res_dic['outside'] = (outside - np.ones(outside.shape)*(img.shape[0]/2.0,
                                                            img.shape[1]/2.0))
    res_dic['img_shape'] = img.shape

    return res_dic

def extract_nuclei(CP_dapi_path, center_inside, raw_inside, img_shape):
    """
    Parameters
    ----------
    CP_dapi_path : string
      path to the csv output file from CellProfiler

    Return
    ----------
    clockwise_centers : np.array of shape (Nf, 2)
      coordinates of the nuclei centers ordonned clockwise.

    """
    dapi_df = pd.read_csv(CP_dapi_path)
    centers = np.column_stack((dapi_df['AreaShape_Center_X'],
                               dapi_df['AreaShape_Center_Y']))

    centers = _delete_artifact(centers, raw_inside)

    clockwise_centers = _arrange_centers_clockwise(centers, center_inside)

    clockwise_centers -= np.full(np.asarray(clockwise_centers).shape,
                                 (img_shape[0]/2.0, img_shape[1]/2.0))
    #delete doubled centers. np.unique does not do the job...
    #use a trick from https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    x = []
    [x.append(tuple(r)) for r in clockwise_centers if tuple(r) not in x]
    clockwise_centers = np.array(x)
    return clockwise_centers


def get_bissecting_vertices(centers, inners, outers, org_center):
    '''
    Parameters
    ----------
    centers : np.ndarray of shape (Nf,2)
      the centers of the nuclei
    inners : np.ndarray
      array describing the inner contour of the organo
    outers : np.ndarray
      array describing the outer contour of the organo
    org_center : tuple (x,y)
      coordinates of the center of the organo

    Returns
    -------
    inner_vs : nd.array of shape (Nf,2)
      the coordinates of the vertices on the inner contour
    outer_vs : nd.array of shape (Nf,2)
      the coordinates of the vertices on the outer contour
    '''
    theta_centers = np.arctan2(centers[:, 1]-org_center[1],
                               centers[:, 0]-org_center[0])
    bissect = (theta_centers + np.roll(theta_centers, 1, axis=0))/2
    dtheta = (theta_centers - np.roll(theta_centers, 1, axis=0))
    # periodic boundary
    bissect[dtheta >= np.pi] -= np.pi
    bissect[dtheta < -np.pi] += np.pi
    theta_inners = np.arctan2(inners[:, 1]-org_center[1],
                              inners[:, 0]-org_center[0])
    theta_outers = np.arctan2(outers[:, 1]-org_center[1],
                              outers[:, 0]-org_center[0])

    inner_vs = inners.take(_find_closer_angle(bissect, theta_inners), axis=0)
    outer_vs = outers.take(_find_closer_angle(bissect, theta_outers), axis=0)
    return inner_vs, outer_vs

def _recognize_in_from_out(retained_contours, centers, radii):
    '''Reliable recognition of the inner and outer contours
    Parameters
    ----------
    centers : np.ndarray of shape (Nf,2)
      the centers of the nuclei
    org_center : tuple (x,y)
      coordinates of the center of the organo

    Returns
    -------
    clockwise_centers : np.array of shape (Nf, 2)
      coordinates of the centers of the nuclei, arrange clockwise
    '''
    retained_contours = np.array((np.squeeze(retained_contours[0]),
                                  np.squeeze(retained_contours[1])))

    dist = retained_contours - \
            np.array((np.full(retained_contours[0].shape, centers[0]),
                      np.full(retained_contours[1].shape, centers[1])))

    norm = np.array((np.mean(np.linalg.norm(dist[0], axis=1)),
                     np.mean(np.linalg.norm(dist[1], axis=1))))
    res = {}
    inside, outside = retained_contours[np.argsort(norm)]
    res['center_inside'], res['center_outside'] = centers[np.argsort(norm)]
    res['rIn'], res['rOut'] = radii[np.argsort(norm)]
    return inside, outside, res

def _arrange_centers_clockwise(centers, org_center):
    '''Arrange the centers clockwise
    Parameters
    ----------
    centers : np.ndarray of shape (Nf,2)
      the centers of the nuclei
    org_center : tuple (x,y)
      coordinates of the center of the organo

    Returns
    -------
    clockwise_centers : np.array of shape (Nf, 2)
      coordinates of the centers of the nuclei, arrange clockwise
    '''
    centers = [(i[0], i[1]) for i in centers]
    #for each cell defined by its center, compute its closest clockwise neighbor
    neighbors = _find_neighbor(centers, org_center)
    #we need the list of cell nuclei to be clockwise ordered
    clockwise_centers = [neighbors[centers[0]]]
    for i in centers[0:len(centers)-1]:
        tmp = clockwise_centers[len(clockwise_centers)-1]
        clockwise_centers.append(neighbors[tmp])
    return clockwise_centers

def _delete_artifact(centers, raw_inside):
    '''
    Parameters
    ----------
    centers : np.ndarray of shape (Nf,2)
      the centers of the nuclei
    raw_inside : np.array
      squeezed contour from opencv of the inner membrane.

    Returns
    -------
    centers : np.array
      the centers of the nuclei that are not artifacts
    '''
    #delete the centers which are not inside the organo (possibly newborn cells ??)
    c2m = np.array([np.min(np.linalg.norm(np.full(raw_inside.shape, center)-
                                          raw_inside, axis=1)) for center in centers])
    to_del = np.argwhere(c2m > 2*np.mean(c2m))
    return np.delete(centers, to_del, 0)

#Computing the nearest clockwise neighbouring nucleus
def _find_neighbor(centers, org_center):
    '''
    Parameters
    ----------
    cells : np.ndarray of shape (Nf,2)
      the centers of the nuclei
    org_center : tuple (x,y)
      coordinates of the center of the organo

    Returns
    -------
    neighbors : dictionnary with fields
      coordinates of the centers of the nuclei
      and values coordinates of the centers of the neighbor of the field key
    '''
    neighbors = {}
    # nested loops are bad and should be avoided
    # you can look into np.meshgrid to have a flat iterator
    # over a 2D grid

    # Here, you should rather look into
    # the scipy.spatial module, and the
    # KDTree neighbor finding structure
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    for i in centers:
        ii = (i[0]-org_center[0], i[1]-org_center[1])
        min1 = 10**6
        argmin1 = -1
        for j in centers:
            jj = (j[0]-org_center[0], j[1]-org_center[1])
            if not ii[0] == jj[0] and not ii[1] == jj[1]:
                dot_prod = ii[0]*jj[1]-ii[1]*jj[0]
                distance = np.sqrt((ii[0]-jj[0])**2 + (ii[1]-jj[1])**2)
                if dot_prod > 0 and distance < min1:
                    min1 = np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)
                    argmin1 = j
        neighbors[i] = argmin1
    return neighbors



def _find_closer_angle(theta0, theta1):
    '''Finds the index of the closest value in theta1
    for each value in theta0, with 2π periodic boundary
    conditions.

    Parameters
    ----------
    theta0 : np.ndarray of shape (N0,)
      the target values
    theta1 : np.ndarray of shape (N1,)
      array where we search for the values closest
      to the targer theta0

    Returns
    -------
    indices : nd.array of shape (N0,)
      the indices of the values closest to theta0 in theta1

    Example
    -------
    >>> theta0 = np.array([0, 0.5, 0.79])*2*np.pi
    >>> theta1 = np.array([0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.])*2*np.pi
    >>> find_closer_angle(theta0, theta1)
        np.array([0, 4, 5])
    '''
    tt0, tt1 = np.meshgrid(theta0, theta1)
    dtheta = tt0 - tt1
    # periodic boundary
    dtheta[dtheta > np.pi] -= 2*np.pi
    dtheta[dtheta <= -np.pi] += 2*np.pi

    return (dtheta**2).argmin(axis=0)

def _fill_gaps(contour, gap_dist):
    '''Finds the gaps in a contour from opencv findContours and
    fill it with a straight line.

    Parameters
    ----------
    contour : np.array of shape (x,1,2)
      the contour from opencv findContours
    gap_dist : float>0
      the distance for two consecutive points in the contour to be
      considered as a gap.

    Returns
    -------
    res : nd.array of shape (y,2)
      the points to ad to the contour so that it is closed.
    '''
    distance = np.linalg.norm(contour-np.roll(contour, -1, axis=0), axis=1)
    gaps = np.squeeze(np.argwhere(distance > gap_dist))
    res = np.empty((0, 2))
    for gap in gaps:
        line = Line(contour[gap], contour[(gap+1)%len(contour)])
        if line.slope == oo:
            xcord = np.full((int(distance[gap]), 1), contour[gap][0])
            ycord = np.linspace(contour[gap][1]+
                                (contour[(gap+1)%len(contour)][1]-
                                 contour[gap][1])/distance[gap],
                                contour[(gap+1)%len(contour)][1],
                                int(distance[gap]), endpoint=False)
            points = np.column_stack((xcord, np.float32(ycord)))
        elif line.slope == 0:
            xcord = np.linspace(contour[gap][0]+
                                (contour[(gap+1)%len(contour)][0]-
                                 contour[gap][0])/distance[gap],
                                contour[(gap+1)%len(contour)][0],
                                int(distance[gap]), endpoint=False)
            ycord = np.full((int(distance[gap]), 1), contour[gap][1])
            points = np.column_stack((xcord, np.float32(ycord)))
        else:
            b = contour[gap][1] - contour[gap][0] * line.slope
            xcord = np.linspace(contour[gap][0]+
                                (contour[(gap+1)%len(contour)][0]-
                                 contour[gap][0])/distance[gap],
                                contour[(gap+1)%len(contour)][0],
                                int(distance[gap]), endpoint=False)
            ycord = int(line.slope) * xcord + b
            points = np.column_stack((xcord, np.float32(ycord)))
        res = np.concatenate((res, points))
    return res
