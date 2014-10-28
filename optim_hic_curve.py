__author__ = 'hervemn'

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.optimize import fsolve
# from scipy.optimize import minimize
from scipy.optimize import leastsq
from leastsqbound import *

def log_residuals(p, y, x):

    d0, d1, alpha_0, alpha_1, alpha_2, A = p
    hic_c = np.zeros(x.shape)
    if d1 > d0:
        if d0 > 0:
            val_lim_0 = np.log(A) + alpha_0 * np.log(d0) - alpha_1 * np.log(d0)
        else:
            val_lim_0 = -10**15.
        val_lim_1 = val_lim_0 + alpha_1 * np.log(d1) - alpha_2 * np.log(d1)
    else:
        val_lim_1 = -10**15
        val_lim_0 = -10**15

    for i in range(0, len(hic_c)):
        if x[i] <= 0:
            hic_c[i] = 0
        elif x[i] <= d0 and x[i] > 0:
            hic_c[i] = np.log(A) + alpha_0 * np.log(x[i])
        elif x[i] > d0 and x[i] <= d1:
            hic_c[i] = val_lim_0 + alpha_1 * np.log(x[i])
        elif x[i] > d1:
            hic_c[i] = val_lim_1 + alpha_2 * np.log(x[i])


    err = y - hic_c
    return err


def log_peval(x, param):

    d0, d1, alpha_0, alpha_1, alpha_2, A = param
    hic_c = np.zeros(x.shape)
    if d1 > d0:
        if d0 > 0:
            val_lim_0 = np.log(A) + alpha_0 * np.log(d0) - alpha_1 * np.log(d0)
        else:
            val_lim_0 = -10**15.
        val_lim_1 = val_lim_0 + alpha_1 * np.log(d1) - alpha_2 * np.log(d1)
    else:
        val_lim_1 = -10**15
        val_lim_0 = -10**15
    for i in range(0, len(hic_c)):
        if x[i] <= 0:
            hic_c[i] = 0
        elif x[i] <= d0 and x[i] > 0:
            hic_c[i] = np.log(A) + alpha_0 * np.log(x[i])
        elif x[i] > d0 and x[i]<= d1:
            hic_c[i] = val_lim_0 + alpha_1 * np.log(x[i])
        elif x[i] > d1:
            hic_c[i] = val_lim_1 + alpha_2 * np.log(x[i])
    return hic_c


def peval(x, param):

    d0, d1, alpha_0, alpha_1, alpha_2, A = param
    hic_c = np.zeros(x.shape)
    if d1 > d0:
        if d0 > 0:
            val_lim_0 = A * np.power(d0, alpha_0) / np.power(d0, alpha_1)
        else:
            val_lim_0 = -10**15.
        val_lim_1 = val_lim_0 * np.power(d1, alpha_1) / np.power(d1, alpha_2)
    else:
        val_lim_0 = -10**15
        val_lim_1 = -10**15

    for i in range(0, len(hic_c)):
        if x[i] <= 0:
            hic_c[i] = 0
        elif x[i] <= d0 and x[i] > 0:
            hic_c[i] = A * np.power(x[i], alpha_0)
        elif x[i] > d0 and x[i] <= d1:
            # hic_c[i] = A * np.power(x[i], alpha_1)
            hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)
        elif x[i] > d1:
            hic_c[i] = val_lim_1 * np.power(x[i], alpha_2)
    return hic_c


def estimate_param_hic(y_meas, x_bins):
    d0 = 20.0
    d1 = 300.0
    alpha_0 = -1.5
    alpha_1 = -1.5
    alpha_2 = -1.5
    x0 = x_bins.min()
    print "x0 = ", x0
    A = np.max(y_meas) * (x0 ** (- alpha_0))
    p0 = [d0, d1, alpha_0, alpha_1, alpha_2, A]
    plsq = leastsq(log_residuals, p0, args=(np.log(y_meas), x_bins))
    print plsq
    # plsq[0][2] = -0.9
    # plsq[0][5] = np.max(y_meas) * (x0 ** (- plsq[0][2]))

    y_estim = peval(np.arange(x_bins.min(), x_bins.max(), 5), plsq[0])

    return plsq, y_estim


def residual_4_max_dist(x, p):
    d0, d1, alpha_0, alpha_1, alpha_2, A, y = p
    hic_c = np.zeros(x.shape)
    if d1 > d0:
        if d0 > 0:
            val_lim_0 = A * np.power(d0, alpha_0) / np.power(d0, alpha_1)
        else:
            val_lim_0 = -10**15.
        val_lim_1 = val_lim_0 * np.power(d1, alpha_1) / np.power(d1, alpha_2)
    else:
        val_lim_0 = -10**15
        val_lim_1 = -10**15
    for i in range(0, len(hic_c)):
        if x[i] <= 0:
            hic_c[i] = 0
        elif x[i] <= d0 and x[i] > 0:
            hic_c[i] = A * np.power(x[i], alpha_0)
        elif x[i] > d0 and x[i] <= d1:
            # hic_c[i] = A * np.power(x[i], alpha_1)
            hic_c[i] = val_lim_0 * np.power(x[i], alpha_1)
        elif x[i] > d1:
            hic_c[i] = val_lim_1 * np.power(x[i], alpha_2)
    err = y - hic_c
    return err


def estimate_max_dist_intra(p, val_inter):
    print "val_inter = ", val_inter

    d0, d1, alpha_0, alpha_1, alpha_2, A = p
    p0 = [d0, d1, alpha_0, alpha_1, alpha_2, A, val_inter]
    s0 = d1
    x = fsolve(residual_4_max_dist, s0, args=(p0))
    print "limit inter/intra distance = ", x
    print "val model @ dist inter = ", peval(x, p)
    # raw_input("alors?")
    return x[0]
