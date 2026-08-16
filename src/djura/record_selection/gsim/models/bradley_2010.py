import numpy as np

from ..base import GMPE
from ..imt import ASI, SA, IMT
from ..contexts import Context
from .. import const
from .gmpe_avgsa import CORRELATION_FUNCTION_HANDLES
from .bradley_2009 import _get_weights, _get_intensity_moments


#: Period range over which ASI is defined, eq. (1)
ASI_PERIOD_RANGE = (0.1, 0.5)


def _get_periods(n_per: int, spacing: str) -> np.ndarray:
    """Discretisation of the 0.1-0.5s period range, p. 795

    T_i = 0.1 + 0.4 * (i - 1) / (n - 1), for linear spacing
    T_i = exp[ln(0.1) + ln(5) * (i - 1) / (n - 1)], for log spacing

    Parameters
    ----------
    n_per : int
        Number of integration points
    spacing : str
        Spacing of the periods, 'linear' or 'log'

    Returns
    -------
    numpy.ndarray
        Vibration periods at which SA is computed

    Raises
    ------
    ValueError
        n_per is less than three, or spacing is neither 'linear' nor 'log'
    """
    t_low, t_high = ASI_PERIOD_RANGE

    if n_per < 3:
        raise ValueError(
            "At least three integration points are required for ASI")

    if spacing == "linear":
        return np.linspace(t_low, t_high, n_per)
    if spacing == "log":
        return np.exp(np.linspace(np.log(t_low), np.log(t_high), n_per))

    raise ValueError(f"Period spacing {spacing} not recognized, "
                     "must be either 'linear' or 'log'")


class Bradley2010ASI(GMPE):

    #: Supported tectonic region type is inherited from the SA model
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {ASI}

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
                 n_per: int = 9, spacing: str = "log", **kwargs):
        """Indirect ground motion model for the acceleration spectrum
        intensity, ASI, originally proposed by Von Thun et al. (1988) and
        defined as the integral of the 5% damped pseudo-spectral
        acceleration between 0.1 and 0.5s, eq. (1). ASI is therefore in
        units of g.s whenever the spectral accelerations of the underlying
        model are in units of g.

        Rather than being calibrated on ASI observations, the median and
        lognormal standard deviation of ASI are computed from the predictions
        of an arbitrary spectral acceleration model and a model for the
        correlation between spectral accelerations at different vibration
        periods. ASI is shown to be adequately represented by a lognormal
        distribution, fig. 1. The formulation is that employed for the
        spectrum intensity by Bradley et al. (2009), and the shared equations
        are those of :mod:`bradley_2009`.

        Parameters
        ----------
        gmpe : GMPE
            Instance of the ground motion model used for the computation of
            the spectral accelerations from which ASI is derived
        corr_func : str, optional
            Handle of the function to compute correlation coefficients between
            different spectral acceleration ordinates. Valid options are:
            'baker_jayaram', 'akkar', 'aristeidou', 'eshm20', 'none'
        n_per : int, optional
            Number of periods used to discretise the 0.1-0.5s period range.
            Nine integration points are found to be appropriate for a wide
            range of magnitude and distance scenarios, fig. 2
        spacing : str, optional
            Spacing of the periods over the 0.1-0.5s period range, 'log' or
            'linear'. Logarithmic spacing converges faster, as the correlation
            between spectral accelerations is a function of the difference in
            the logarithm of their periods

        Raises
        ------
        ValueError
            n_per or spacing are not valid, or corr_func is not a valid
            correlation function
        """
        self.gmpe_name = gmpe

        # Combine the parameters of the GMPE provided at the construction
        # level with the ones assigned to the ASI GMPE.
        for key in dir(self):
            if key.startswith('REQUIRES_'):
                setattr(self, key, getattr(self.gmpe_name, key))
            if key.startswith('DEFINED_'):
                if not key.endswith('FOR_INTENSITY_MEASURE_TYPES'):
                    setattr(self, key, getattr(self.gmpe_name, key))

        self.periods = _get_periods(n_per, spacing)
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
        acceleration spectrum intensity, ASI

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
        Bradley, B. A. (2010). Site-Specific and Spatially Distributed
        Ground-Motion Prediction of Acceleration Spectrum Intensity.
        Bulletin of the Seismological Society of America, 100(2), 792-801.
        https://doi.org/10.1785/0120090157
        """
        mean_sa, sig_sa, phi_sa = self._get_sa_predictions(ctx)
        median_sa = np.exp(mean_sa)
        rho_ln = self.corr_func.rho

        # Total-event median and dispersion of ASI, eq. (7) and (8)
        median, sig = _get_intensity_moments(
            median_sa, sig_sa, self.weights, rho_ln)
        mean = np.log(median)

        if phi_sa is None:
            return mean, [np.array([sig])]

        # The intra-event dispersion of ASI is obtained in the same way as
        # the total-event dispersion, with the intra-event dispersion of SA
        # adopted in eq. (2), (3) and (6). The correlation of the intra-event
        # residuals is very similar to that of the total residuals, and the
        # same correlation model is therefore used
        _, phi = _get_intensity_moments(
            median_sa, phi_sa, self.weights, rho_ln)
        tau = np.sqrt(np.clip(sig ** 2. - phi ** 2., 0., None))

        return mean, [
            np.array([sig]),
            np.array([tau]),
            np.array([phi])
        ]
