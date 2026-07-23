"""Calculation of a one-sided tolerance interval for a normal distribution.

Formula comes from the Engineering Statistics Handbook Section 7.2.6.3 "Tolerance intervals for a normal distribution"
https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm.
"""

import logging

from numpy import mean, std, sqrt
from scipy.stats import norm

logger = logging.getLogger(__name__)


def compute_k1(n, prop, confidence):
    """Compute k1 factor."""
    assert n > 2
    dof = n - 1
    # We want the ISF for the lower tail, so use (1 - prop)
    gauss_critical = norm.isf(1 - prop)
    logger.debug(f'Gaussian critical value: {gauss_critical:.3f} (coverage={prop*100:.2f}%)')
    gauss_critical2 = norm.isf(confidence)
    a = 1 - gauss_critical2 ** 2 / (2 * dof)
    b = gauss_critical ** 2 - gauss_critical2 ** 2 / n
    k1 = (gauss_critical + sqrt(gauss_critical ** 2 - (a * b))) / a
    return k1


def compute_lower_tolerance(data, prop, confidence):
    """Compute lower tolerance.

    Parameters
    ----------
    data : array of float
        The data for computation.
    prop : float
        The percentage of the population that should be above the lower limit.
    confidence : float
        The confidence value - percentage (between 0 and 1).
    verbose : bool
        Enable logging, by default True.

    Returns
    -------
    float
        The 100 PPM accuracy value.
    """
    k1 = compute_k1(len(data), prop, confidence)
    data_mean = mean(data)
    data_std = std(data)
    lower = data_mean - k1 * data_std
    logger.debug(f'{lower:.4f} covers {prop*100:.3f}% of data with a confidence of {confidence*100:.2f}%')
    return lower
