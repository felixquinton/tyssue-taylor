"""Model specific functions and class defintitions
"""
import numpy as np
import pandas as pd
from tyssue import PlanarGeometry
from tyssue.dynamics import effectors, units
from tyssue.generation import generate_ring
from tyssue.dynamics.factory import model_factory


def mesh_from_data(centers, inner_contour, outer_contour):
    """Creates an annular organoid from image data
    """

    Nf = centers.shape[0]

    # Organize centers clockwize
    origin = centers.mean(axis=1)
    shifted_centers = centers - origin[np.newaxis, :]
    thetas = np.arctan2(shifted_centers[:, 1],
                        shifted_centers[:, 0])
    centers = centers.take(np.argsort(thetas))
    inner_vs, outer_vs = get_bissecting_vertices(centers,
                                                 inner_contour,
                                                 outer_contour)

    R_in = np.linalg.norm(inner_vs - inner_vs.mean(axis=1), axis=0).mean()
    R_out = np.linalg.norm(outer_vs - outer_vs.mean(axis=1), axis=0).mean()

    organo = generate_ring(Nf, R_in, R_out)
    organo.vert_df.loc[organo.apical_verts, organo.coords] = inner_vs[::-1]
    organo.vert_df.loc[organo.basal_verts, organo.coords] = outer_vs[::-1]

    AnnularGeometry.update_all(organo)
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area': organo.face_df.area.mean(), #and there was an error here
            'area_elasticity': 1,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'fx': 0.,
            'fy': 0.,
            'sx': 0., # source and target coordinates
            'sy': 0.,
            'tx': 0.,
            'ty': 0.,
			'line_tension': 1e-3,
            'is_active': 1
            },
        'vert':{
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 10,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    return organo



def find_closer_angle(theta0, theta1):
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
    dtheta[dtheta >   np.pi] -= 2*np.pi
    dtheta[dtheta <= -np.pi] += 2*np.pi

    return (dtheta**2).argmin(axis=0)

def get_bissecting_vertices(centers, inner_contour, outer_contour):
    '''Docstring left as an exercice
    '''
    theta_centers = np.arctan2(centers[:, 1], centers[:, 0])
    bissect = (theta_centers + np.roll(theta_centers, 1, axis=0))/2
    dtheta = (theta_centers - np.roll(theta_centers, 1, axis=0))
    # periodic boundary
    bissect[dtheta >= np.pi] -= np.pi
    bissect[dtheta < -np.pi] += np.pi
    theta_inners = np.arctan2(inner_contour[:, 1],
                              inner_contour[:, 0])
    theta_outers = np.arctan2(outer_contour[:, 1],
                              outer_contour[:, 0])

    inner_vs = inner_contour.take(
        find_closer_angle(bissect, theta_inners), axis=0)
    outer_vs = outer_contour.take(
        find_closer_angle(bissect, theta_outers), axis=0)
    return inner_vs, outer_vs



# The following classes will probably be included in tyssue at some point
class AnnularGeometry(PlanarGeometry):
    """
    """
    @classmethod
    def update_all(cls, eptm):
        PlanarGeometry.update_all(eptm)
        cls.update_lumen_volume(eptm)

    @staticmethod
    def update_lumen_volume(eptm):
        srce_pos = eptm.upcast_srce(eptm.vert_df[['x', 'y']]).loc[eptm.apical_edges]
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[['x', 'y']]).loc[eptm.apical_edges]
        apical_edge_pos = (srce_pos + trgt_pos)/2
        apical_edge_coords = eptm.edge_df.loc[eptm.apical_edges,
                                              ['dx', 'dy']]
        eptm.settings['lumen_volume'] = (
            - apical_edge_pos['x'] * apical_edge_coords['dy']
            + apical_edge_pos['y'] * apical_edge_coords['dx']).values.sum()


class BasalAdhesion(effectors.AbstractEffector):
    """

    """
    dimensions = units.line_elasticity
    label = 'Basal adhesion to the ECM'
    magnitude = 'adhesion_strength'
    element = 'vert'
    spatial_ref = 'x', units.length

    specs = {
        'vert': {
            'adhesion_strength', # put to zero where relevant
            'x_ecm', 'y_ecm', # To be set at initialisation
            'x', 'y',
            }
        }

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval('0.5 * adhesion_strength '
                                 '* ((x - x_ecm)**2 '
                                 '+  (y - y_ecm)**2)')

    @staticmethod
    def gradient(eptm):

        grad_x = eptm.vert_df.eval('adhesion_strength * (x - x_ecm)')
        grad_y = eptm.vert_df.eval('adhesion_strength * (y - y_ecm)')
        grad = pd.DataFrame({'gx': grad_x, 'gy': grad_y})
        return grad, None



class LumenElasticity(effectors.AbstractEffector):
    '''

    .. math:: \frac{K_Y}{2}(A_{\mathrm{lumen}} - A_{0,\mathrm{lumen}})^2

    '''
    dimensions = units.area_elasticity
    label = 'Lumen volume constraint'
    magnitude = 'lumen_elasticity'
    element = 'settings'
    spatial_ref = 'lumen_prefered_vol', units.area

    specs = {
        'settings': {
            'lumen_elasticity',
            'lumen_prefered_vol',
            'lumen_volume'
            }
        }

    @staticmethod
    def energy(eptm):
        Ky = eptm.settings['lumen_elasticity']
        V0 = eptm.settings['lumen_prefered_vol']
        Vy = eptm.settings['lumen_volume']
        return np.array([Ky * (Vy - V0)**2 / 2,])

    @staticmethod
    def gradient(eptm):
        Ky = eptm.settings['lumen_elasticity']
        V0 = eptm.settings['lumen_prefered_vol']
        Vy = eptm.settings['lumen_volume']
        grad_srce, grad_trgt = lumen_area_grad(eptm)
        return (Ky*(Vy - V0) * grad_srce,
                Ky*(Vy - V0) * grad_trgt)


def lumen_area_grad(eptm):
    apical_pos = eptm.vert_df[['x', 'y']].copy()
    apical_pos.loc[eptm.basal_verts] = 0
    srce_pos = eptm.upcast_srce(apical_pos)
    trgt_pos = eptm.upcast_trgt(apical_pos)
    grad_srce = srce_pos.copy()
    grad_srce.columns = ['gx', 'gy']
    grad_trgt = grad_srce.copy()
    grad_srce['gx'] = trgt_pos['y']
    grad_srce['gy'] = -trgt_pos['x']
    grad_trgt['gx'] = -srce_pos['y']
    grad_trgt['gy'] = srce_pos['x']
    # minus sign due to the backward orientation
    return -grad_srce, -grad_trgt

model = model_factory([effectors.FaceAreaElasticity,
                       effectors.LineTension,
                       BasalAdhesion,
                       LumenElasticity],
                      effectors.FaceAreaElasticity)
