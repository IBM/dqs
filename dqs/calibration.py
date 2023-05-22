import numpy as np

import sys
sys.path.append('.')
from qrdr import kaplan_meier

def Dcalibration(quantiles, e, boundaries):
    '''
    Compute D-Calibration.

    Parameters
    ----------
    quantiles : ndarray
        Quantiles of observed times (event times and censoring times).
        Tensor shape is [batch_size].
    e : ndarray
        Indicator (censored (0) or uncensored (1))
        Tensor shape is [batch_size].
    boundaries: list(double)
        Bin boundaries used to compute D-calibration
    Returns
    -------
    D-Calibration : Tensor
        A non-negative float number.
    '''
    uncensored = e.astype(np.bool)
    len_bin = boundaries[1:] - boundaries[:-1]
    num_bin = len(boundaries)-1

    # compute count_unc for uncensored data points
    count_unc = np.zeros((quantiles[uncensored].shape[0], num_bin))
    t = quantiles[uncensored].reshape(-1,1)
    t_in_C = ((boundaries[:-1] <= t) & (t < boundaries[1:]))
    count_unc[t_in_C] += 1.0
    count_unc[:,-1] = 1.0 - np.sum(count_unc[:,:-1], 1)

    # compute count_cen for censored data points
    count_cen = np.zeros((quantiles[~uncensored].shape[0], num_bin))
    v = quantiles[~uncensored].reshape(-1,1)
    v_in_C = ((boundaries[:-1] <= v) & (v < boundaries[1:]))
    count_cen[v_in_C] += ((boundaries[1:] - v) / (1-v))[v_in_C]
    v_leq_C = (v < boundaries[:-1])
    count_cen[v_leq_C] += (len_bin / (1-v))[v_leq_C]
    count_cen[:,-1] = 1.0 - np.sum(count_cen[:,:-1], 1)

    # compute square loss
    diff = (np.sum(count_unc,0)+np.sum(count_cen,0))/quantiles.shape[0]
    diff -= len_bin
    return np.sum(diff * diff)

def KMcalibration(f_pred, z, e, boundaries, EPS=0.000001):
    '''
    Compute KM-Calibration
    Parameters
    ----------
    f_pred : ndarray
        Prediction results with n_bin+1 endpoints.
        Each row corresponds to a prediction.
        sum_i F_pred[:,i] = 1.0
        Tensor shape is [batch_size, n_bin].
    z : ndarray
        Observation time (event time or censored time)
        Tensor shape is [batch_size].
    e : ndarray
        Indicator (censored (0) or uncensored (1))
        Tensor shape is [batch_size].
    boundaries : list (float)
        Boundaries of f_pred
        Each element in z must be STRICTLY smaller than boundaries[-1]
    Returns
    -------
    KM-Calibration : Tensor
        A non-negative float number.
    '''

    e_dist, invalid_idx = kaplan_meier.estimate_dist(z, e, boundaries)
    f_pred_mean = np.mean(f_pred,0)

    # compute logarithmic loss for KM valid region
    log_empirical = np.log(e_dist[:invalid_idx]+EPS)
    log_mean_pred = np.log(f_pred_mean[:invalid_idx]+EPS)
    loss_valid = np.sum(e_dist[:invalid_idx]*(log_empirical-log_mean_pred))

    # compute logarithmic loss for KM invalid region
    sum_empirical = np.sum(e_dist[invalid_idx:])
    log_sum_empirical = np.log(sum_empirical + EPS)
    log_sum_pred = np.log(np.sum(f_pred_mean[invalid_idx:]) + EPS)
    loss_invalid = sum_empirical*(log_sum_empirical-log_sum_pred)

    return loss_valid + loss_invalid
