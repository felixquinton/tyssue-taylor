"""This module contains functions that are used to display various info on an
organoid.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from tyssue.generation import generate_ring
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.draw.plt_draw import sheet_view

from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
from tyssue_taylor.segmentation.segment2D import normalize_scale
from tyssue_taylor.adjusters.adjust_annular import (set_init_point,
                                                    prepare_tensions)


def rendering_results(organo, x_data, y_data, title, xlabel, ylabel, legend):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data,
                                                                   y_data)
    fig, ax = plt.subplots()
    plt.plot(x_data, y_data, '.', markersize=10, alpha=0.4)
    plt.plot(x_data, intercept+slope*np.array(x_data), '-')
    plt.title(title, fontdict={'fontsize': 32})
    plt.legend(legend, loc='upper left', fontsize=16)
    plt.xlabel(xlabel, fontdict={'fontsize': 24})
    plt.ylabel(ylabel, fontdict={'fontsize': 24})
    fig.set_size_inches(12, 12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    std = np.sum(np.power(intercept+slope*np.array(x_data)-y_data, 2))
    print('R value :', r_value,
          '\nStandard error :', (std/organo.Ne)**0.5)


def rendering_convergence_results(x_data, y_data, title, xlabel, ylabel,
                                  legend, data_dot='-', rol_win=50):
    fig, ax = plt.subplots()
    plt.plot(x_data, y_data, data_dot, markersize=10, alpha=0.4)
    rolling = y_data.rolling(rol_win, min_periods=0, center=True).mean()
    plt.plot(x_data, rolling, data_dot, markersize=20, alpha=1)
    plt.title(title, fontdict={'fontsize': 32})
    plt.legend(legend, loc='upper left', fontsize=16)
    plt.xlabel(xlabel, fontdict={'fontsize': 24})
    plt.ylabel(ylabel, fontdict={'fontsize': 24})
    fig.set_size_inches(12, 12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def print_tensions(exp_organo, th_organo):
    draw_specs = sheet_spec()
    tension_max = np.max(exp_organo.edge_df.line_tension.values.copy())
    edge_color = 1/tension_max*exp_organo.edge_df.line_tension.values.copy()
    cmap = plt.cm.get_cmap('viridis')
    edge_cmap = cmap(edge_color)
    draw_specs['vert']['visible'] = False
    draw_specs['edge']['color'] = edge_cmap
    draw_specs['edge']['width'] = 0.25+3*edge_color
    fig, ax = quick_edge_draw(th_organo, lw=5, c='k', alpha=0.2)
    fig, ax = sheet_view(exp_organo, ax=ax, **draw_specs)
    fig.set_size_inches(12, 12)
    plt.xlabel('Size in µm')
    plt.ylabel('Size in µm')


def plot_force_inference(organo, coefs, cmap='tab20'):
    draw_specs = sheet_spec()
    edge_color = np.ones(organo.Nf*3)
    cmap = plt.cm.get_cmap(cmap)
    edge_cmap = cmap(edge_color)
    draw_specs['vert']['visible'] = False
    draw_specs['edge']['color'] = edge_cmap
    draw_specs['edge']['width'] = edge_cmap
    fig, ax = sheet_view(organo, **draw_specs)
    U = np.concatenate([coefs[i][coefs[i] != 0]
                        for i in range(organo.Nv)])
    V = np.concatenate([coefs[i+6][coefs[i+6] != 0]
                        for i in range(organo.Nv)])
    X = np.concatenate([[organo.vert_df.x[i]]*len(coefs[i][coefs[i] != 0])
                        for i in range(organo.Nv)])
    Y = np.concatenate([[organo.vert_df.y[i]]*len(coefs[i][coefs[i] != 0])
                        for i in range(organo.Nv)])
    C = np.concatenate([np.argwhere(coefs[i] != 0)
                        for i in range(organo.Nv)])
    for i in range(organo.Nv):
        q = ax.quiver(X, Y, U, V, C, cmap=cmap)
        ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                     label='Quiver key, length = 10', labelpos='E')
    fig.set_size_inches(12, 12)
    plt.xlabel('Size in µm')
    plt.ylabel('Size in µm')
