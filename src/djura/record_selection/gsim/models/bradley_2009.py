import numpy as np

from ..base import GMPE
from ..imt import SI, SA, IMT
from ..contexts import Context
from .. import const
from .gmpe_avgsa import CORRELATION_FUNCTION_HANDLES


#: Period range over which SI is defined, eq. (1)
SI_PERIOD_RANGE = (0.1, 2.5)

#: Gravitational acceleration in [cm/s2], used to convert the spectral
#: accelerations of the underlying model from units of g
GRAVITY = 981.


def _get_periods(delta_t: float) -> np.ndarray:
    """Discretisation of the 0.1-2.5s period range, eq. (7)

    Parameters
    ----------
    delta_t : float
        Size of the vibration period discretisation, i.e. the step-size
        used in the integration, in [s]

    Returns
    -------
    numpy.ndarray
        Vibration periods at which SA is computed

    Raises
    ------
    ValueError
        delta_t is not positive, or is larger than the period range
    """
    t_low, t_high = SI_PERIOD_RANGE

    if delta_t <= 0. or delta_t > (t_high - t_low) / 2.:
        raise ValueError(
            f"Discretisation step {delta_t} is not valid, at least three "
            f"integration points are required over the {t_low}-{t_high}s "
            "period range")

    # The number of integration steps is rounded up so that the period range
    # is always integrated over in full, with the step-size reduced to the
    # nearest divisor of the period range whenever it is not one already
    n_steps = int(np.ceil((t_high - t_low) / delta_t - 1e-9))

    return np.linspace(t_low, t_high, n_steps + 1)


def _get_weights(periods: np.ndarray) -> np.ndarray:
    """Trapezoidal integration weights of eq. (7)

    The weights take the value of 0.5 for the first and last periods and
    1.0 otherwise, scaled by the discretisation step. They are expressed
    here in the equivalent form which also holds for a non-constant step

    w_1 = (T_2 - T_1) / 2
    w_n = (T_n - T_n-1) / 2
    w_i = (T_i+1 - T_i-1) / 2

    Parameters
    ----------
    periods : numpy.ndarray
        Vibration periods at which SA is computed

    Returns
    -------
    numpy.ndarray
        Integration weights
    """
    weights = np.zeros_like(periods)
    weights[0] = (periods[1] - periods[0]) / 2.
    weights[-1] = (periods[-1] - periods[-2]) / 2.
    weights[1:-1] = (periods[2:] - periods[:-2]) / 2.

    return weights


def _get_pseudo_spectral_velocity(median_sa, periods):
    """Median pseudo-spectral velocity, eq. (4)

    PSV(T_i) = SA(T_i) / omega_i = SA(T_i) * T_i / (2 * pi)

    As the conversion is a deterministic scaling of SA, the lognormal
    standard deviations and the correlations of PSV are those of SA,
    eq. (5), (6) and (A1)

    Parameters
    ----------
    median_sa : numpy.ndarray
        Median spectral accelerations in units of g, of shape
        (n_per, n_sites)
    periods : numpy.ndarray
        Vibration periods at which SA is computed, of shape (n_per, )

    Returns
    -------
    numpy.ndarray
        Median pseudo-spectral velocities in [cm/s]
    """
    return median_sa * GRAVITY * periods[:, None] / (2. * np.pi)


def _get_nonlog_moments(median, sigma_ln):
    """Nonlog moments of the spectral accelerations, eq. (2) and (3)

    mu_SA = SA_50 * exp(0.5 * sigma_lnSA ** 2)
    sigma_SA = mu_SA * sqrt(exp(sigma_lnSA ** 2) - 1)

    Parameters
    ----------
    median : numpy.ndarray
        Median (fiftieth percentile) values
    sigma_ln : numpy.ndarray
        Lognormal standard deviations

    Returns
    -------
    numpy.ndarray and numpy.ndarray
        Nonlog means and standard deviations
    """
    mu = median * np.exp(0.5 * sigma_ln ** 2.)
    sigma = mu * np.sqrt(np.exp(sigma_ln ** 2.) - 1.)

    return mu, sigma


def _get_nonlog_correlation(rho_ln, sigma_ln_i, sigma_ln_j):
    """Correlation between the nonlog spectral accelerations, eq. (10)

    rho = (exp(rho_ln * sig_i * sig_j) - 1)
        / (sqrt(exp(sig_i ** 2) - 1) * sqrt(exp(sig_j ** 2) - 1))

    The first order approximation of eq. (10), rho ~ rho_ln, is not
    adopted here, as the exact expression is used throughout the article
    unless otherwise stated

    Parameters
    ----------
    rho_ln : numpy.ndarray
        Correlation between the logarithms of the spectral accelerations
        at periods T_i and T_j
    sigma_ln_i : numpy.ndarray
        Lognormal standard deviation of the spectral acceleration at T_i
    sigma_ln_j : numpy.ndarray
        Lognormal standard deviation of the spectral acceleration at T_j

    Returns
    -------
    numpy.ndarray
        Correlation between the nonlog spectral accelerations
    """
    return (np.exp(rho_ln * sigma_ln_i * sigma_ln_j) - 1.) / (
        np.sqrt(np.exp(sigma_ln_i ** 2.) - 1.)
        * np.sqrt(np.exp(sigma_ln_j ** 2.) - 1.))


def _get_intensity_moments(median, sigma_ln, weights, rho_ln):
    """Median and lognormal standard deviation of the integral of a
    response spectrum, eq. (8), (9), (11) and (12)

    mu = sum_i(w_i * mu_i)
    sigma ** 2 = sum_i(sum_j(w_i * w_j * rho_ij * sig_i * sig_j))
    IM_50 = mu ** 2 / sqrt(sigma ** 2 + mu ** 2)
    sigma_lnIM = sqrt(ln((sigma / mu) ** 2 + 1))

    Parameters
    ----------
    median : numpy.ndarray
        Median spectral ordinates, of shape (n_per, n_sites)
    sigma_ln : numpy.ndarray
        Lognormal standard deviations of the spectral ordinates, of shape
        (n_per, n_sites)
    weights : numpy.ndarray
        Integration weights, of shape (n_per, )
    rho_ln : numpy.ndarray
        Correlation matrix of the logarithms of the spectral ordinates,
        of shape (n_per, n_per)

    Returns
    -------
    numpy.ndarray and numpy.ndarray
        Medians and lognormal standard deviations of the intensity measure
    """
    # Nonlog moments of the spectral ordinates, eq. (2) and (3)
    mu_im, sigma_im = _get_nonlog_moments(median, sigma_ln)

    # Mean of the intensity measure, eq. (8)
    mu = np.sum(weights[:, None] * mu_im, axis=0)

    # Variance of the intensity measure, eq. (9), with the correlation
    # between the nonlog spectral ordinates obtained from eq. (10)
    rho = _get_nonlog_correlation(
        rho_ln[:, :, None], sigma_ln[:, None, :], sigma_ln[None, :, :])
    var = np.sum(
        weights[:, None, None] * weights[None, :, None] * rho
        * sigma_im[:, None, :] * sigma_im[None, :, :], axis=(0, 1))

    # Median and dispersion of the intensity measure, eq. (11) and (12)
    median_im = mu ** 2. / np.sqrt(var + mu ** 2.)
    sigma_ln_im = np.sqrt(np.log(var / mu ** 2. + 1.))

    return median_im, sigma_ln_im


class BradleyEtAl2009SI(GMPE):

    #: Supported tectonic region type is inherited from the SA model
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {SI}

    #: Supported intensity measure components are inherited from the SA model
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''

    #: Supported standard deviation types
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}

    #: Required sites parameters are inherited from the SA model
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameters are inherited from the SA model
    REQUIRES_RUPTURE_PARAMETERS = set()

    #: Required distance measures are inherited from the SA model
    REQUIRES_DISTANCES = set()

    #: No independent tests - verification against paper
    non_verified = True

    def __init__(self, gmpe, corr_func: str = "baker_jayaram",
                 delta_t: float = 0.1, **kwargs):
        """Indirect ground motion model for the spectrum intensity, SI,
        originally proposed by Housner (1952, 1963) and defined as the
        integral of the 5% damped pseudo-spectral velocity between 0.1 and
        2.5s, eq. (1). SI is therefore in units of cm.s/s whenever the
        spectral accelerations of the underlying model are in units of g.

        Rather than being calibrated on SI observations, the median and
        lognormal standard deviation of SI are computed from the predictions
        of an arbitrary spectral acceleration model and a model for the
        correlation between spectral accelerations at different vibration
        periods. SI is shown to be adequately represented by a lognormal
        distribution, over both the body and the tails of the distribution,
        fig. 1 and fig. 3.

        Parameters
        ----------
        gmpe : GMPE
            Instance of the ground motion model used for the computation of
            the spectral accelerations from which SI is derived
        corr_func : str, optional
            Handle of the function to compute correlation coefficients between
            different spectral acceleration ordinates. Valid options are:
            'baker_jayaram', 'akkar', 'aristeidou', 'eshm20', 'none'
        delta_t : float, optional
            Size of the vibration period discretisation in [s]. A step-size
            below 0.2s is appropriate for a wide range of magnitude and
            distance scenarios, fig. 4

        Raises
        ------
        ValueError
            delta_t is not valid, or corr_func is not a valid correlation
            function
        """
        self.gmpe_name = gmpe

        # Combine the parameters of the GMPE provided at the construction
        # level with the ones assigned to the SI GMPE.
        for key in dir(self):
            if key.startswith('REQUIRES_'):
                setattr(self, key, getattr(self.gmpe_name, key))
            if key.startswith('DEFINED_'):
                if not key.endswith('FOR_INTENSITY_MEASURE_TYPES'):
                    setattr(self, key, getattr(self.gmpe_name, key))

        self.periods = _get_periods(delta_t)
        self.weights = _get_weights(self.periods)

        # Check for existing correlation function
        if corr_func not in CORRELATION_FUNCTION_HANDLES:
            raise ValueError('Not a valid correlation function')
        else:
            self.corr_func = CORRELATION_FUNCTION_HANDLES[corr_func](
                self.periods)

    def _get_sa_predictions(self, ctx: Context):
        """Computes the spectral accelerations at each of the integration
        periods through the underlying ground motion model

        Parameters
        ----------
        ctx : Context
            Instance of Context class which contains the site parameters and,
            rupture and site-to-source distance parameters of the scenario.

        Returns
        -------
        numpy.ndarray
            Means of the logarithms of the spectral accelerations, of shape
            (n_per, n_sites)
        numpy.ndarray
            Total lognormal standard deviations, of shape (n_per, n_sites)
        numpy.ndarray or None
            Intra-event lognormal standard deviations, of shape
            (n_per, n_sites), None if not provided by the underlying model
        """
        mean, sig, phi = [], [], []

        for period in self.periods:
            params = self.gmpe_name.get_mean_and_stddevs(ctx, SA(period))
            stddevs = params[1]

            mean.append(np.atleast_1d(np.squeeze(params[0])))
            sig.append(np.atleast_1d(np.squeeze(stddevs[0])))

            if len(stddevs) == 3:
                phi.append(np.atleast_1d(np.squeeze(stddevs[2])))

        return np.array(mean), np.array(sig), np.array(phi) if phi else None

    def get_mean_and_stddevs(self, ctx: Context, imt: IMT):
        """Provides the ground motion prediction equation for the
        spectrum intensity, SI

        Parameters
        ----------
        ctx : Context
            Instance of Context class which contains the site parameters and,
            rupture and site-to-source distance parameters of the scenario.
        imt : IMT
            Instance of IMT class which describes the intensity measure type.

        Returns
        -------
        numpy.ndarray and float
            Means and stadard deviations

        Reference
        -------
        Bradley, B. A., Cubrinovski, M., MacRae, G. A., & Dhakal, R. P.
        (2009). Ground-Motion Prediction Equation for SI Based on Spectral
        Acceleration Equations. Bulletin of the Seismological Society of
        America, 99(1), 277-285. https://doi.org/10.1785/0120080044
        """
        mean_sa, sig_sa, phi_sa = self._get_sa_predictions(ctx)
        rho_ln = self.corr_func.rho

        # Median pseudo-spectral velocities, eq. (4)
        median_psv = _get_pseudo_spectral_velocity(
            np.exp(mean_sa), self.periods)

        # Total-event median and dispersion of SI, eq. (11) and (12)
        median, sig = _get_intensity_moments(
            median_psv, sig_sa, self.weights, rho_ln)
        mean = np.log(median)

        if phi_sa is None:
            return mean, [np.array([sig])]

        # The intra-event dispersion of SI is obtained in the same way as
        # the total-event dispersion, with the intra-event dispersion of SA
        # adopted in eq. (2), (3) and (10). The correlation of the intra-event
        # residuals is very similar to that of the total residuals, and the
        # same correlation model is therefore used
        _, phi = _get_intensity_moments(
            median_psv, phi_sa, self.weights, rho_ln)
        tau = np.sqrt(np.clip(sig ** 2. - phi ** 2., 0., None))

        return mean, [
            np.array([sig]),
            np.array([tau]),
            np.array([phi])
        ]
