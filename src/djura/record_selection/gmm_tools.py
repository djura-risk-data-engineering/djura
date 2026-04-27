import numpy as np


def calculate_epsilon(im_star_value, mean, sigma):
    """Back-calculates epsilon

    Parameters
    ----------
    im_star_value : float
        Value of IM*
    mean : Union[float, np.ndarray]
        Mean predictions of IM*
    sigma : Union[float, np.ndarray]
        Standard deviations of predictions of IM*

    Returns
    -------
    Union[float, np.ndarray]
        Epsilon values
    """
    return (np.log(im_star_value) - mean) / sigma


def get_sigma_xy(sigma, corr):
    """Calculate sigma correlated to IM*

    Parameters
    ----------
    sigma : Union[float, np.ndarray]
        Stdev of IMi conditioned on rupture scenario
    corr : np.ndarray
        Correlations between IMi and IM*

    Returns
    -------
    np.ndarray
        Stdevs correlated to IM*
    """
    if not isinstance(sigma, float) and corr.shape != sigma.shape:
        corr = corr.reshape(-1, 1).flatten()

    return sigma * np.sqrt(1 - corr ** 2)


def get_mean_xy(mean, sigma, corr, epsilon):
    """Compute mean_{ln(IMi|IM*,rup)} for each GMM case of IM*

    Parameters
    ----------
    mean : np.ndarray
        Means of IMi conditioned on rupture scenario
    sigma : np.ndarray
        Stdev of IMi conditioned on rupture scenario
    corr : np.ndarray
        Correlations between IMi and IM*
    epsilon : np.ndarray
        Epsilon_{lnIM*|rup}

    Returns
    -------
    np.ndarray
        Mean_{ln(IMi|IM*,rup)}
    """

    if not isinstance(sigma, float) and corr.shape != sigma.shape:
        corr = corr.reshape(-1, 1).flatten()

    mean_cond = mean + corr * sigma * epsilon

    return mean_cond.T
