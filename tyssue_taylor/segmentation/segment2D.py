"""Image segmentation module
This module provides utilities to extract nuclei and membranes
contours from microscopy images
"""
from glob import glob
import numpy as np
import pandas as pd
from scipy import interpolate as intplt
from tyssue.generation import generate_ring
from tifffile import imread
from stardist import dist_to_coord, non_maximum_suppression, StarDist
from csbdeep.utils import normalize
import cv2 as cv


def generate_ring_from_image(brightfield_path, dapi_path,
                             scp_model_path=None,
                             threshold=28, blur=9,
                             rol_window_inside=100,
                             rol_window_outside=20):
    """Create an organo mesh of class AnnularSheet from a brightfield image
    and a CellProfiler DAPI analysis csv file

    Parameters
    ----------
    brightfield_path : string
      path to the brightfield image
    dapi_path : string
      path to the DAPI image
    scp_model_path : string
      path to the stardist model
    threshold: int >=0
      threshold to apply to the brightfield image
    blur: int >=0
      gaussian blur to apply to the brightfield image
    rol_window_inside: int > 0.
      Length of the window of the rolling mean on apical contour.
    rol_window_outside: int > 0.
      Length of the window of the rolling mean on basal contour.
    Return
    ----------
    organo : object of class AnnularSheet
      the organo mesh extracted from the data
    """

    membrane_dic = extract_membranes(brightfield_path, threshold, blur)
    clockwise_centers = _star_convex_polynoms(dapi_path,
                                              membrane_dic,
                                              scp_model_path)

    inside_df = pd.DataFrame(membrane_dic['inside'], index=None)
    outside_df = pd.DataFrame(membrane_dic['outside'], index=None)
    inners = inside_df.rolling(rol_window_inside, min_periods=1).mean().values
    outers = outside_df.rolling(rol_window_outside, min_periods=1).mean().values

    nb_cells = len(clockwise_centers)

    org_center = membrane_dic['center_inside'] - \
        np.full(2, membrane_dic['img_shape'][0]/2.0)

    inner_vs, outer_vs = get_bissecting_vertices(clockwise_centers, inners,
                                                 outers, org_center)

    organo = generate_ring(nb_cells, membrane_dic['rIn'], membrane_dic['rOut'])

    organo.vert_df.loc[organo.apical_verts,
                       organo.coords] = (inner_vs[::-1]-np.full(
                           inner_vs.shape, org_center))*0.323
    organo.vert_df.loc[organo.basal_verts,
                       organo.coords] = (outer_vs[::-1]-np.full(
                           outer_vs.shape, org_center))*0.323
    inners = (inners-np.full(inners.shape, org_center))*0.323
    outers = (outers-np.full(outers.shape, org_center))*0.323
    clockwise_centers = np.array(clockwise_centers)
    clockwise_centers -= np.full(outer_vs.shape, org_center)
    clockwise_centers *= 0.323
    organo.settings['R_in'] *= 0.323
    organo.settings['R_out'] *= 0.323

    return organo, inners, outers, clockwise_centers


def _star_convex_polynoms(dapi_path, membrane_dic, model_path):
    """Nuclei segmentation using the stardist algorithm.

    Parameters
    ----------
    dapi_path : string
      path to the DAPI image
    membrane_dic: dic obtained with extract_membranes.
      Dictionnary containing relevant informations about the membranes.
    model_path : string
      path to the stardist model
    Return
    ----------
    clockwise_centers : np.ndarray
      An array of cell centers, sort in clockwise order.
    """
    images = sorted(glob(dapi_path))
    images = list(map(imread, images))
    img = normalize(images[0], 1, 99.8)
    model_sc = StarDist(None,
                        name='stardist_shape_completion',
                        basedir=model_path)
    prob, dist = model_sc.predict(img)
    coord = dist_to_coord(dist)
    points = non_maximum_suppression(coord, prob, prob_thresh=0.4)
    points = np.flip(points, 1)

    rho, phi = _card_coords(points, membrane_dic['center_inside'])
    cleaned = _quick_del_art(points, rho, membrane_dic['rIn'])
    clockwise_centers = _quick_clockwise(cleaned,
                                         phi, rho,
                                         membrane_dic['rIn'])

    clockwise_centers = np.subtract(np.float32(clockwise_centers),
                                    np.array(membrane_dic['img_shape'])/2.0)

    return clockwise_centers


def extract_membranes(brightfield_path, threshold=2, blur=9):
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
    res_dic : dic
      Dictionnary with items:
      - raw_inside : apical contour coordinates in the image.
      - raw_outside : basal contour coordinates in the image.
      - inside : apical contour coordinates centered on (0, 0).
      - outside : basal contour coordinates centered on (0, 0).
      - img_shape : shape of the brightfield image.
    """
    img = cv.imread(brightfield_path, cv.IMREAD_GRAYSCALE).copy()
    _, img = cv.threshold(img, threshold, 255, 0)
    img = cv.GaussianBlur(img, (blur, blur), 0)

    img, contours, _ = cv.findContours(img, cv.RETR_TREE,
                                       cv.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)

    contours_length = np.array([c.size for c in contours])
    membrane_ind = np.argsort(contours_length)[-2:]
    if threshold == 2:
        membrane_ind = np.argsort(contours_length)[::2]

    retained_contours = contours[membrane_ind]

    circles = np.array((cv.minEnclosingCircle(retained_contours[0]),
                        cv.minEnclosingCircle(retained_contours[1])))

    centers = circles[:, 0]
    radii = circles[:, 1]
    inside, outside, res_dic = _recognize_in_from_out(retained_contours,
                                                      centers, radii)

    inside = np.concatenate((inside, _fill_gaps(inside, 2)))
    outside = np.concatenate((outside, _fill_gaps(outside, 2)))

    res_dic['raw_inside'] = inside
    res_dic['raw_outside'] = outside

    res_dic['inside'] = (inside - np.ones(inside.shape)*(img.shape[0]/2.0,
                                                         img.shape[1]/2.0))
    res_dic['outside'] = (outside - np.ones(outside.shape)*(img.shape[0]/2.0,
                                                            img.shape[1]/2.0))
    res_dic['img_shape'] = img.shape

    return res_dic


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

    bissect[dtheta >= np.pi] -= np.pi
    bissect[dtheta < -np.pi] += np.pi
    theta_inners = np.arctan2(inners[:, 1]-org_center[1],
                              inners[:, 0]-org_center[0])
    theta_outers = np.arctan2(outers[:, 1]-org_center[1],
                              outers[:, 0]-org_center[0])

    inner_vs = inners.take(_find_closer_angle(bissect, theta_inners), axis=0)
    outer_vs = outers.take(_find_closer_angle(bissect, theta_outers), axis=0)
    return inner_vs, outer_vs


def normalize_scale(organo, geom, refer='area'):
    """Rescale an organo so that the mean cell area is close to 1.
    Useful if one as some issues with the scale of the optimization parameters.
    Parameters
    ----------
    organo : :class:`Epithelium` object
      the organo to rescale

    geom : tyssue geometry class

    refer : string
      if 'area': coordinates are changed so that the mean cell area is 1.
      if 'edges': coordinates are changed so that the mean edge length is 1.
      Default is 'area'.

    Return
    ----------
    res_organo : :class:`Epithelium` object
      the rescaled organo

    """
    old_area = organo.face_df.area.copy()
    old_lumen_vol = organo.settings['lumen_volume']
    if refer == 'area':
        mean = organo.face_df.area.mean()
        organo.vert_df.loc[:, organo.coords] /= mean**0.5
        organo.settings['R_in'] /= mean**0.5
        organo.settings['R_out'] /= mean**0.5
        if 'adhesion_strength' in organo.vert_df.columns:
            organo.vert_df.loc[:, ('x_ecm', 'y_ecm')] /= mean
    elif refer == 'edges':
        mean = organo.edge_df.length.mean()
        organo.vert_df.loc[:, organo.coords] /= mean
        organo.settings['R_in'] /= mean
        organo.settings['R_out'] /= mean
        if 'adhesion_strength' in organo.vert_df.columns:
            organo.vert_df.loc[:, ('x_ecm', 'y_ecm')] /= mean
    geom.update_all(organo)
    area_factor = organo.face_df.area/old_area
    organo.face_df.prefered_area *= area_factor
    organo.settings['lumen_prefered_vol'] *= (organo.settings['lumen_volume'] /
                                              old_lumen_vol)
    return organo


def _recognize_in_from_out(retained_contours, centers, radii):
    '''Recognition of the inner and outer contours
    Parameters
    ----------
    retained_contours : list of contours
      Two contours that are candidates for inner and outer membranes.
    centers : tuple of size 2
      the centers of the wo contours
    radii : tuple of size 2
      The radii of the two contours

    Returns
    -------
    clockwise_centers : np.array of shape (Nf, 2)
      coordinates of the centers of the nuclei, arrange clockwise
    '''
    retained_contours = np.array((retained_contours[0].squeeze(),
                                  retained_contours[1].squeeze()))

    rho0, phi0 = _card_coords(retained_contours[0], centers[0])
    rho1, phi1 = _card_coords(retained_contours[1], centers[1])
    sort_ind = np.argsort((rho0.sum(), rho1.sum()))

    res = {}
    inside, outside = retained_contours[sort_ind]
    res['center_inside'], res['center_outside'] = centers[sort_ind]
    res['rIn'], res['rOut'] = radii[sort_ind]
    return inside, outside, res


def _card_coords(array, center):
    """Convert cartesian coordinates into polar coordinates.
    Parameters
    ----------
    array : np.ndarray of shape (x, 2)
      An array of 2d points
    center : 2d point
      The point at the center on the polar coordinates system.

    Return
    ----------
    rho : np.ndarray of shape x
      the table of radii
    phi : np.ndarray of shape x
      the table of angles
    """
    x, y = array[:, 0], array[:, 1]
    x0, y0 = center
    rho = np.linalg.norm(np.c_[x-x0, y-y0], axis=1)
    phi = np.arctan2(y-y0, x-x0)
    return rho, phi


def _quick_clockwise(array, phi, rho, radius):
    """Sort an array on points in clockwise order.
    Parameters
    ----------
    array : np.ndarray of shape (x, 2)
      An array of 2d points
    phi : np.ndarray of shape x
      angle elements of the polar coordinates of array
    rho : np.ndarray of shape x
      radius elements of the polar coordinates of array
    radius: float.
      the radius of the ring.

    Return
    ----------
    rho : np.ndarray of shape x
      the table of radii
    phi : np.ndarray of shape x
      the table of angles
    """
    res = array[np.argsort(phi[rho > radius*0.8]), :]
    return res


def _quick_del_art(array, rho, radius):
    """Deletes cells that are too far inside the organoid.
    Parameters
    ----------
    array : np.ndarray of shape (x, 2)
      An array of 2d points
    rho : np.ndarray of shape x
      radius elements of the polar coordinates of array
    radius: float.
      the radius of the ring.

    Return
    ----------
    res : np.ndarray
      Array of points that are on the ring.
    """
    res = array[rho > radius*0.8]
    return res


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
    dtheta[dtheta > np.pi] -= 2*np.pi
    dtheta[dtheta <= -np.pi] += 2*np.pi

    return (dtheta**2).argmin(axis=0)


def _fill_gaps(contour, gap_dist):
    ''' Finds the gaps in a contour from opencv findContours and
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
    gaps = np.argwhere(distance > gap_dist).squeeze()
    res = np.empty((0, 2))
    for gap in gaps:
        x = contour[gap]
        y = contour[(gap+1) % len(contour)]
        pts = np.vstack(((x[0], y[0]), (x[1], y[1])))
        tck, u = intplt.splprep(pts, k=1)
        u_new = np.linspace(u.min(), u.max(), 10)
        x_new, y_new = intplt.splev(u_new, tck, der=0)
        res = np.concatenate((res,
                              np.column_stack((x_new, y_new))))
    return res
