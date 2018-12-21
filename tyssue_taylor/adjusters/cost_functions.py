"""This module provides cost functions and constraints for the Optimization
process
"""
import warnings
import numpy as np

def distance_regularized(eptm, objective_eptm, variables,
                         solver, geom, model,
                         to_regularize={}, reg_weight=0,
                         sum_residuals=False,
                         coords=None, **kwargs):
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
    """
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
    reg_mod = reg_weight * _reg_module(tmp_eptm,
                                       to_regularize.get('apical', False),
                                       to_regularize.get('basal', False))
    dist = np.linalg.norm(_distance(tmp_eptm,
                                    objective_eptm, coords), axis=1)
    tension_bound = _tension_bounds(tmp_eptm)
    if sum_residuals:
        #obj = dist + np.sum(np.concatenate((np.array(reg_mod),
        #                                    tension_bound.values)))
        obj = np.sum(np.concatenate((dist, reg_mod, tension_bound)))
    else:
        obj = np.concatenate((dist, reg_mod, tension_bound))
    #print(obj)
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
    #norm = np.linalg.norm(diff, axis=1)
    return diff

def _reg_module(actual_eptm, reg_apical, reg_basal):
    apical_edges = actual_eptm.edge_df.loc[actual_eptm.apical_edges].copy()
    apical_module = np.square(apical_edges.line_tension-
                              np.roll(apical_edges.line_tension, -1))
    basal_edges = actual_eptm.edge_df.loc[actual_eptm.basal_edges].copy()
    basal_module = np.square(basal_edges.line_tension -
                             np.roll(basal_edges.line_tension, -1))
    return np.concatenate((int(reg_apical)*np.asarray(apical_module),
                           int(reg_basal)*np.asarray(basal_module)))

def _tension_bounds(actual_eptm, coords=None):
    if coords is None:
        coords = actual_eptm.coords
    tensions = actual_eptm.edge_df.loc[:, 'line_tension'][:3*actual_eptm.Nf].copy()
    tension_lb = -np.minimum(tensions, np.zeros(3*actual_eptm.Nf))
    tension_ub = np.zeros(3*actual_eptm.Nf)
    tension_ub[tensions > 1e3] = tensions[tensions > 1e3] - 1e3
    return 1e3*np.power((tension_lb + tension_ub),
                        np.full(tension_lb.shape, 3))
