import numpy as np


def estimate_curve(z, e):
    '''
    Compute Kaplan-Meier curve

    Parameters
    ----------
    z: array of size N
        event time or last observation time in the dataset
    e: array of size N
        censored (0) or uncensored (1)

    Returns
    -------
    time_points: numpy.ndarray
        array of times, where each entry corresponds an uncensored event time
        empty numpy.ndarray if all data points are censored
    pi: numpy.ndarray
        array of probabilities
        empty numpy.ndarray if all data points are censored
    '''

    # inner function
    def count_num_survive(time_point, t):
        return np.count_nonzero(t >= time_point)

    # get list of uncensored event times
    time_points, counts = np.unique(z[e > 0], return_counts=True)
    if len(time_points) == 0:
        return time_points, counts

    # compute survival rate of each time by vectorize
    vfunc = np.vectorize(count_num_survive)
    vfunc.excluded.add(1)  # fix second argument
    ni = vfunc(time_points, z)
    p = 1 - counts/ni
    pi = np.cumprod(p)

    return time_points, pi

def estimate_dist(z, e, boundaries, EPS=0.0000001):
    '''
    Estimate empirical distribution by using Kaplan-Meier estimator.
    Note:
    Suppose that a patient dies at time 10
    and another patient dies at time 20.
    Let [0, 10), [10, 20), [20, 30) be time intervals.
    Then we have
      survival_rate[time 0] = 1
      survival_rate[time 10] = 0.5
      survival_rate[time 20] = 0
    Hence
      empirical distribution in time 0-10 = 0.5
      empirical distribution in time 10-20 = 0.5
      empirical distribution in time 20-30 = 0.0
    This definition is OK, because we want to train a prediction model
    such that
      survival_rate[time 0] = 1
      survival_rate[time 10] = 0.5
      survival_rate[time 20] = 0
    If we define
      empirical distribution in time 0-10 = 0.0
      empirical distribution in time 10-20 = 0.5
      empirical distribution in time 20-30 = 0.5
    then the prediction model is trained to learn that
      survival_rate[time 0] = 1
      survival_rate[time 10] = 1
      survival_rate[time 20] = 0.5
    This problem is mostly related to the uncontinous nature of
    Kaplan-Meier curve.  For uncensored time t, the Kaplan-Meier curve
    has two valid data points due to the uncontinuous curve.
    We take the lower value for such data points.

    Parameters
    ----------
    z : ndarray (float)
        Array of event times or last observation times.
        Tensor shape is [batch size].
    e : ndarray (bool)
        Array indicating censored (0) or uncensored (1).
        Tensor shape is [batch_size].
    boundaries : list (float)
        Each element in z must be STRICTLY less than boundaries[-1].

    Returns
    -------
    empirical_dist : Tensor
        Empirical distribution of size [n_bin]
    time_KM_invalid_index : int
        Fast index of the time which estimation is invalid.
        The first censored data point after the last uncensored
        data point (or the last uncensored data point if such
        data point is none) is considered last valid data point.
        empirical_dist[:time_KM_invalid_index] contains valid estimation.
        0 <= time_KM_invalid_index < n_bin by definition.
        This means that empirical_dist[-1] is always considered invalid.
    '''

    # estimate KM curve
    uncensored = e.astype(np.bool)
    all_censored = False
    time_points_KM, survival_rates_KM = estimate_curve(z,e)
    if len(time_points_KM) == 0:  # if all data points are censored
        time_points_KM = np.array([ z.min() ])
        all_censored = True
    if time_points_KM[0] == 0.0:
        time_points_KM[0] = EPS
    survival_rates_KM = np.append(1.0, survival_rates_KM)
    survival_rates_KM = np.append(survival_rates_KM, 0.0)

    # compute valid index
    z_censored = z[~uncensored]
    temp = z_censored[z_censored > time_points_KM[-1]]
    if all_censored or len(temp)==0:
        invalid_index = np.searchsorted(boundaries,
                                        time_points_KM[-1],
                                        side='right')
    else:
        invalid_index = np.searchsorted(boundaries,
                                        temp.min(),
                                        side='right')
    invalid_index -= 1

    # compute emprirical distribution from KM curve
    indices_tp_threshold = np.searchsorted(time_points_KM,
                                           boundaries,
                                           side='right')
    survival_rates = np.take(survival_rates_KM, indices_tp_threshold)
    survival_rates[-1] = 0.0
    empirical_dist = survival_rates[:-1] - survival_rates[1:]
    return empirical_dist, invalid_index
