"""This module provides cost functions and constraints for the Optimization
process
"""
import warnings
import numpy as np
import csv
from tyssue_taylor.models.annular import AnnularGeometry as geom


def distance_regularized(eptm, objective_eptm, variables,
                         solver, geom, model,
                         to_regularize={}, reg_weight=0,
                         sum_residuals=False,
                         coords=None,
                         IPRINT=None,
                         COPY_OR_SYM='copy',
                         **kwargs):
    """Changes variables inplace in the epithelium, finds the energy minimum,
    and returns the distance between the new configuration and the objective
    epithelium.
    Parameters
    ----------
    eptm : :class:`Epithelium` object
    objective_eptm : :class:`Epithelium` object to refer to
    variables : dict of values to be changed (see bellow)
    solver : solver to find energy minimum
    geom : tyssue geometry class
    model : tyssue dynamic model
    coords : list of strings over which the distance to the objective
      is computed if `None`, will default to the eptm.coords
    **kwargs : keyword arguments passed to the energy optimisation
    variables is a dict of values. The keys are a pair with the element to
    change i.e.
    IPRINT : string. Path to a file used to save the value of the distance.
    """
    if COPY_OR_SYM == 'copy':
        tmp_eptm = eptm.copy()
    elif COPY_OR_SYM == 'sym':
        tmp_eptm = create_organo(eptm.Nf,
                                 eptm.settings['R_in'],
                                 eptm.settings['R_out'])
    for (elem, columns), values in variables.items():
        if elem in tmp_eptm.data_names:
            tmp_eptm.datasets[elem][columns] = values
        elif elem in tmp_eptm.settings:
            tmp_eptm.settings[elem] = values
    res = solver.find_energy_min(tmp_eptm, geom, model, **kwargs)
    if not res.get('success', True):
        warnings.warn('Energy optimisation failed')
        print(res.get('message', 'no message was provided'))
    reg_mod = reg_weight * _reg_module(tmp_eptm,
                                       to_regularize.get('apical', False),
                                       to_regularize.get('basal', False))
    dist = np.linalg.norm(_distance(tmp_eptm,
                                    objective_eptm, coords), axis=1)
    tension_bound = _tension_bounds(tmp_eptm)
    if sum_residuals:
        # obj = dist + np.sum(np.concatenate((np.array(reg_mod),
        #                                    tension_bound.values)))
        obj = np.sum(np.concatenate((dist, reg_mod, tension_bound)))
    else:
        obj = np.concatenate((dist, reg_mod, tension_bound))
    # print(obj)
    if IPRINT is not None:
        with open(IPRINT, mode='a+') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            if not sum_residuals:
                csv_writer.writerow([np.sum(
                    np.concatenate((dist, reg_mod, tension_bound)))])
            else:
                csv_writer.writerow([obj])
    print(np.sum(np.concatenate((dist, reg_mod, tension_bound))),
          obj)
    return obj


def _energy(eptm, variables, solver, geom, model, **kwargs):
    tmp_eptm = eptm.copy()
    for (elem, columns), values in variables.items():
        if elem in tmp_eptm.data_names:
            tmp_eptm.datasets[elem][columns] = values
        elif elem in tmp_eptm.settings:
            tmp_eptm.settings[elem] = values
    res = solver.find_energy_min(tmp_eptm, geom, model, **kwargs)
    if not res.get('success', True):
        warnings.warn('Energy optimisation failed')
        print(res.get('message', 'no message was provided'))
    return model.compute_energy(tmp_eptm)


def _distance(actual_eptm, objective_eptm, coords=None):
    if coords is None:
        coords = objective_eptm.coords
    diff = np.subtract(np.array(actual_eptm.vert_df[coords].values),
                       np.array(objective_eptm.vert_df[coords].values))
    # norm = np.linalg.norm(diff, axis=1)
    return diff


def _reg_module(actual_eptm, reg_apical, reg_basal):
    apical_edges = actual_eptm.edge_df.loc[actual_eptm.apical_edges].copy()
    apical_module = np.square(apical_edges.line_tension -
                              np.roll(apical_edges.line_tension, -1))
    basal_edges = actual_eptm.edge_df.loc[actual_eptm.basal_edges].copy()
    basal_module = np.square(basal_edges.line_tension -
                             np.roll(basal_edges.line_tension, -1))
    return np.concatenate((int(reg_apical)*np.asarray(apical_module),
                           int(reg_basal)*np.asarray(basal_module)))


def _tension_bounds(actual_eptm, coords=None):
    if coords is None:
        coords = actual_eptm.coords
    tensions = (actual_eptm.edge_df.loc[:, 'line_tension']
                [:3*actual_eptm.Nf].copy())
    tension_lb = -np.minimum(tensions, np.zeros(3*actual_eptm.Nf))
    tension_ub = np.zeros(3*actual_eptm.Nf)
    tension_ub[tensions > 1e3] = tensions[tensions > 1e3] - 1e3
    return 1e3*np.power((tension_lb + tension_ub),
                        np.full(tension_lb.shape, 3))


def create_organo(nb_cells, r_in, r_out, seed=None, rot=None, geom=geom):
    organo = generate_ring(nb_cells, r_in, r_out)
    Nf = organo.Nf
    geom.update_all(organo)
    alpha = 1 + 1/(20*(organo.settings['R_out']-organo.settings['R_in']))
    specs = {
        'face': {
            'is_alive': 1,
            'prefered_area': organo.face_df.area,
            'area_elasticity': 1., },
        'edge': {
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.1,
            'is_active': 1
            },
        'vert': {
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.1,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    normalize_scale(organo, geom, refer='edges')
    geom.update_all(organo)
    if seed is not None:
        symetric_tensions = set_init_point(organo.settings['R_in'],
                                           organo.settings['R_out'],
                                           organo.Nf, alpha)
        sin_mul = 1+(np.sin(np.linspace(0, 2*np.pi, organo.Nf,
                                        endpoint=False)))**2
        organo.face_df.prefered_area *= np.random.normal(1.0, 0.05, organo.Nf)
        organo.edge_df.line_tension = prepare_tensions(organo,
                                                       symetric_tensions)
        organo.edge_df.loc[:Nf-1, 'line_tension'] *= sin_mul*np.random.normal(
            1.0, 0.05, organo.Nf)
        geom.update_all(organo)
    if rot is not None:
        organo.vert_df.loc[:, 'x'] = (organo.vert_df.x.copy() * np.cos(rot) -
                                      organo.vert_df.y.copy() * np.sin(rot))
        print('rotated x',
              organo.vert_df.x.copy() * np.cos(rot) -
              organo.vert_df.y.copy() * np.sin(rot))
        organo.vert_df.loc[:, 'y'] = (organo.vert_df.x.copy() * np.sin(rot) +
                                      organo.vert_df.y.copy() * np.cos(rot))
        print('rotated y',
              organo.vert_df.x.copy() * np.sin(rot) +
              organo.vert_df.y.copy() * np.cos(rot))
        geom.update_all(organo)
    organo.vert_df[['x_ecm', 'y_ecm']] = organo.vert_df[['x', 'y']]
    organo.vert_df.loc[organo.basal_verts, 'adhesion_strength'] = 0.01
    new_tensions = organo.edge_df.line_tension
    organo.edge_df.loc[:, 'line_tension'] = new_tensions
    return organo
