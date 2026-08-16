import numpy as np

from ..base import GMPE
from ..imt import DSI, SA, IMT
from ..contexts import Context
from .. import const
from .gmpe_avgsa import CORRELATION_FUNCTION_HANDLES
from .bradley_2009 import (
    GRAVITY, _get_weights, _get_intensity_moments)
from .bradley_2010 import _get_periods


#: Period range over which DSI is defined, eq. (1)
DSI_PERIOD_RANGE = (2.0, 5.0)


def _get_spectral_displacement(median_sa, periods):
    """Median spectral displacement, eq. (4)

    Sd(T_i) = SA(T_i) * (T_i / (2 * pi)) ** 2

    As the conversion is a deterministic scaling of SA, the lognormal
    standard deviations and the correlations of Sd are those of SA,
    eq. (5) and (6)

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
        Median spectral displacements in [cm]
    """
    return median_sa * GRAVITY * (periods[:, None] / (2. * np.pi)) ** 2.


class Bradley2011DSI(GMPE):

    #: Supported tectonic region type is inherited from the SA model
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {DSI}

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
        """Indirect ground motion model for the displacement spectrum
        intensity, DSI, defined as the integral of the 5% damped
        displacement response spectrum between 2.0 and 5.0s, eq. (1), and
        proposed as an indicator of the severity of the long period content
        of a ground motion. DSI is therefore in units of cm.s whenever the
        spectral accelerations of the underlying model are in units of g.

        Rather than being calibrated on DSI observations, the median and
        lognormal standard deviation of DSI are computed from the predictions
        of an arbitrary spectral acceleration model and a model for the
        correlation between spectral accelerations at different vibration
        periods. DSI is shown to be adequately represented by a lognormal
        distribution, fig. 1. The formulation is that employed for the
        spectrum intensity by Bradley et al. (2009), and the shared equations
        are those of :mod:`bradley_2009`.

        Parameters
        ----------
        gmpe : GMPE
            Instance of the ground motion model used for the computation of
            the spectral accelerations from which DSI is derived
        corr_func : str, optional
            Handle of the function to compute correlation coefficients between
            different spectral acceleration ordinates. Valid options are:
            'baker_jayaram', 'akkar', 'aristeidou', 'eshm20', 'none'
        n_per : int, optional
            Number of periods used to discretise the 2.0-5.0s period range.
            Nine integration points are found to be appropriate for a wide
            range of magnitude and distance scenarios, fig. 3
        spacing : str, optional
            Spacing of the periods over the 2.0-5.0s period range, 'log' or
            'linear'. The two provide similar convergence rates and are
            essentially identical in the resulting prediction, fig. 3

        Raises
        ------
        ValueError
            n_per or spacing are not valid, or corr_func is not a valid
            correlation function
        """
        self.gmpe_name = gmpe

        # Combine the parameters of the GMPE provided at the construction
        # level with the ones assigned to the DSI GMPE.
        for key in dir(self):
            if key.startswith('REQUIRES_'):
                setattr(self, key, getattr(self.gmpe_name, key))
            if key.startswith('DEFINED_'):
                if not key.endswith('FOR_INTENSITY_MEASURE_TYPES'):
                    setattr(self, key, getattr(self.gmpe_name, key))

        self.periods = _get_periods(*DSI_PERIOD_RANGE, n_per, spacing)
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
        displacement spectrum intensity, DSI

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
        Bradley, B. A. (2011). Empirical equations for the prediction of
        displacement spectrum intensity and its correlation with other
        intensity measures. Soil Dynamics and Earthquake Engineering,
        31(8), 1182-1191. https://doi.org/10.1016/j.soildyn.2011.04.007
        """
        mean_sa, sig_sa, phi_sa = self._get_sa_predictions(ctx)
        rho_ln = self.corr_func.rho

        # Median spectral displacements, eq. (4)
        median_sd = _get_spectral_displacement(
            np.exp(mean_sa), self.periods)

        # Total-event median and dispersion of DSI, eq. (7) and (8)
        median, sig = _get_intensity_moments(
            median_sd, sig_sa, self.weights, rho_ln)
        mean = np.log(median)

        if phi_sa is None:
            return mean, [np.array([sig])]

        # The intra-event dispersion of DSI is obtained in the same way as
        # the total-event dispersion, with the intra-event dispersion of SA
        # adopted in eq. (2), (3) and (6). The correlation of the intra-event
        # residuals is very similar to that of the total residuals, and the
        # same correlation model is therefore used
        _, phi = _get_intensity_moments(
            median_sd, phi_sa, self.weights, rho_ln)
        tau = np.sqrt(np.clip(sig ** 2. - phi ** 2., 0., None))

        return mean, [
            np.array([sig]),
            np.array([tau]),
            np.array([phi])
        ]
