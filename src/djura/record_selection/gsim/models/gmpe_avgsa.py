# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import abc
import json
from typing import Union
from pathlib import Path
import numpy as np
from ..base import GMPE
from ..imt import AVGSA, SA, IMT
from .. import const
from ..contexts import Context
from ...correlation_models import baker_jayaram, akkar
from ...constants import ESHM20_COEFFICIENTS


asset_dir = Path(__file__).resolve().parents[2] / "assets"
AKKAR_CORRELATION_TABLE = "akkar_correlation_table.txt"


class GmpeIndirectAvgSA(GMPE):
    """Implements an alternative form of GMPE for indirect Average SA (AvgSA)
    that allows for AvgSA to be defined as a vector quantity described by an
    anchoring period (T0) and a set of n_per spectral accelerations linearly
    spaced between t_low * T0 and t_high * T0. This corresponds to the more
    common definition of AvgSA as the mean between, for example, 0.2 * T0 and
    1.5 * T0, used by (among others) Iacoletti et al. (2023).

    In this form of AvgSA GMPE it is possible to run analysis for multiple
    values of AvgSA with different T0 values, such as one might need if
    considering risk for a heterogeneous portfolio of buildings. To do so
    the set of required periods needed for all of the T0 values are assembled
    and SA determined for each of the values needed. However, if the total
    number of SA periods exceeds a user-configurable limit (max_num_per,
    defaulted to 30) then SA will be calculated for the maximum number of
    periods and interpolated to the desired values for each AvgSA(T0).

    :param string gmpe_name:
        The name of a GMPE class used for the calculation.

    :param string corr_func:
        Handle of the function to compute correlation coefficients between
        different spectral acceleration ordinates. Valid options are:
        'baker_jayaram', 'akkar', 'eshm20', 'none'. Default is none.

    :param float t_low:
        Lower bound of period range for calculation (as t_low * T0)

    :param float t_high:
        Upper bound of period range for calculation (as t_high * T0)

    :param int n_per:
        Number of linearly spacee periods beteen t_low * T0 and t_high * T0
        from which AvgSA(T0) is determined

    :param int max_num_per:
        Maximum number of periods permissible for direct calculation of
        AvgSA before switching to an interpolation approach
    """

    # Parameters
    REQUIRES_SITES_PARAMETERS = set()
    REQUIRES_DISTANCES = set()
    REQUIRES_RUPTURE_PARAMETERS = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = ''
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {AVGSA, }
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
    DEFINED_FOR_TECTONIC_REGION_TYPE = ''

    def __init__(self, gmpe, corr_func, t_low: float = 0.2,
                 t_high: float = 2.0, n_per: int = 10, **kwargs):

        self.gmpe_name = gmpe
        # Combine the parameters of the GMPE provided at the construction
        # level with the ones assigned to the average GMPE.
        for key in dir(self):
            if key.startswith('REQUIRES_'):
                setattr(self, key, getattr(self.gmpe_name, key))
            if key.startswith('DEFINED_'):
                if not key.endswith('FOR_INTENSITY_MEASURE_TYPES'):
                    setattr(self, key, getattr(self.gmpe_name, key))

        # Ensure that it is always recogised that the AvgSA GMPE is defined
        # only for total standard deviation even if the called GMPE is
        # defined for inter- and intra-event standard deviations too
        self.DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
        assert t_high > t_low, \
            "Upper bound scaling factor for AvgSA must exceed lower bound"
        self.t_low = t_low
        self.t_high = t_high
        self.t_num = n_per
        self.max_num_per = kwargs.get("max_num_per", 30)

        # Check for existing correlation function
        if corr_func not in CORRELATION_FUNCTION_HANDLES:
            raise ValueError('Not a valid correlation function')
        else:
            self.corr_func = CORRELATION_FUNCTION_HANDLES[corr_func]

    def get_mean_and_stddevs(self, ctx: Context, imt: IMT):

        periods = np.linspace(self.t_low * imt.period,
                              self.t_high * imt.period,
                              self.t_num)

        if len(periods) > self.max_num_per:
            # Maximum number of periods exceeded, so now define a set of
            # max_num_per periods linearly spaced between the lower and
            # upper bound of the total period range considered
            periods = np.linspace(periods[0], periods[-1], self.max_num_per)
            apply_interpolation = True
        else:
            apply_interpolation = False

        # Get mean and stddevs for all required periods
        new_imts = [SA(per) for per in periods]
        mean_sa = np.zeros(len(new_imts))
        sigma_sa = np.zeros_like(mean_sa)

        for imt_i, new_imt in enumerate(new_imts):
            params = self.gmpe_name.get_mean_and_stddevs(ctx, new_imt)
            try:
                mean_i = params[0][0]
                sigma_i = params[1][0][0]
            except IndexError:
                mean_i = params[0]
                sigma_i = params[1]
            mean_sa[imt_i] = mean_i
            sigma_sa[imt_i] = sigma_i

        if apply_interpolation:
            from scipy.interpolate import interp1d

            # Interpolate mean and sigma to the t_num selected periods
            target_periods = np.linspace(self.t_low * imt.period,
                                         self.t_high * imt.period,
                                         self.t_num)

            ipl_mean = interp1d(
                periods, mean_sa.T, bounds_error=False,
                fill_value=(mean_sa[0, :], mean_sa[-1, :]),
                assume_sorted=True)
            mean = ((1.0 / self.t_num) * np.sum(
                ipl_mean(target_periods).T, axis=0))

            ipl_sig = interp1d(
                periods, sigma_sa.T, bounds_error=False,
                fill_value=(sigma_sa[0, :], sigma_sa[-1, :]),
                assume_sorted=True)
            sig_target = ipl_sig(target_periods).T
        else:
            # For the given IM simply select the mean and sigma for the
            # corresponding periods
            target_periods = periods
            mean = (1.0 / self.t_num) * np.sum(mean_sa, axis=0)
            sig_target = sigma_sa

        # For the total standard deviation sum the standard deviations
        # accounting for the cross-correlation
        sig = 0
        for j, t_1 in enumerate(target_periods):
            for k, t_2 in enumerate(target_periods):
                rho = 1.0 if j == k else self.corr_func.get_correlation(
                    t_1, t_2)
                sig += (rho * sig_target[j] * sig_target[k])
        sig = np.sqrt((1.0 / (self.t_num ** 2.)) * sig)

        return np.array([mean]), [np.array([sig])]


class BaseAvgSACorrelationModel(metaclass=abc.ABCMeta):
    """
    Base class for correlation models used in spectral period averaging.
    """

    def __init__(self, avg_periods):
        self.avg_periods = avg_periods
        self.build_correlation_matrix()

    def build_correlation_matrix(self):
        pass

    def __call__(self, i, j):
        return self.rho[i, j]


class BakerJayaramCorrelationModel(BaseAvgSACorrelationModel):
    """
    Produce inter-period correlation for any two spectral periods.
    Subroutine taken from: https://usgs.github.io/shakemap/shakelib
    Based upon:
    Baker, J.W. and Jayaram, N., 2007, Correlation of spectral acceleration
    values from NGA ground motion models, Earthquake Spectra.
    """

    def build_correlation_matrix(self):
        """
        Constucts the correlation matrix period-by-period from the
        correlation functions
        """
        self.rho = np.eye(len(self.avg_periods))
        for i, t1 in enumerate(self.avg_periods):
            for j, t2 in enumerate(self.avg_periods[i:]):
                self.rho[i, i + j] = self.get_correlation(t1, t2)
        self.rho += (self.rho.T - np.eye(len(self.avg_periods)))

    @staticmethod
    def get_correlation(t1, t2):
        """
        Computes the correlation coefficient for the specified periods.

        :param float t1:
            First period of interest.

        :param float t2:
            Second period of interest.

        :return float rho:
            The predicted correlation coefficient.
        """
        return baker_jayaram(t1, t2)


class AkkarCorrelationModel(BaseAvgSACorrelationModel):
    """
    Read the period-dependent correlation coefficient matrix as in:
    Akkar S., Sandikkaya MA., Ay BO., 2014, Compatible ground-motion
    prediction equations for damping scaling factors and vertical to
    horizontal spectral amplitude ratios for the broader Europe region,
    Bull Earthquake Eng, 12, pp. 517-547.
    """

    def build_correlation_matrix(self):
        """
        Constructs the correlation matrix by two-step linear interpolation
        from the correlation table
        """
        from scipy.interpolate import interp1d

        with open(asset_dir / AKKAR_CORRELATION_TABLE, 'r') as file:
            content = file.read()

        iper = np.array([
            0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14,
            0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3,
            0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3,
            1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4,
            3.6, 3.8, 4
        ])

        irho = np.fromstring(
            content, dtype=float, sep=" ").reshape(-1, len(iper))

        if np.any(self.avg_periods < iper[0]) or\
                np.any(self.avg_periods > iper[-1]):
            raise ValueError("'avg_periods' contains values outside of the "
                             "range supported by the Akkar et al. (2014) "
                             "correlation model")
        ipl1 = interp1d(iper, irho, axis=1)
        ipl2 = interp1d(iper, ipl1(self.avg_periods), axis=0)
        self.rho = ipl2(self.avg_periods)

    @staticmethod
    def get_correlation(t1, t2):
        """
        Computes the correlation coefficient for the specified periods.

        :param float t1:
            First period of interest.:w


        :param float t2:
            Second period of interest.

        :return float:
            The predicted correlation coefficient.
        """
        return akkar(t1, t2)


class ESHM20CorrelationModel(BakerJayaramCorrelationModel):
    """Variation of the Baker & Jayaram (2007) cross-correlation model with
    coefficients calibrated on European data, and with separate functions
    for correlation in between-event, between-site and within-event residuals
    """

    @staticmethod
    def get_correlation(t1, t2):
        """
        Computes the correlation coefficient for the specified periods for the
        total standard deviation

        :param float t1:
            First period of interest.

        :param float t2:
            Second period of interest.

        :return float rho:
            The predicted correlation coefficient.

        Original
        0.366, 0.105, 0.0099
        New
        0.20698079,  0.0888577,  -0.03330
        """
        d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["total"]
        return baker_jayaram(t1, t2, d1, d2, d3, d4, d5)

    @staticmethod
    def get_between_event_correlation(t1, t2):
        """As per the get_correlation function but for the between-event
        residuals only
        """
        d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["between-event"]
        return baker_jayaram(t1, t2, d1, d2, d3, d4, d5)

    @staticmethod
    def get_between_site_correlation(t1, t2):
        """As per the get_correlation function but for the between-site
        residuals only
        """
        d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["between-site"]
        return baker_jayaram(t1, t2, d1, d2, d3, d4, d5)

    @staticmethod
    def get_within_event_correlation(t1, t2):
        """As per the get_correlation function but for the between-event
        residuals only
        """
        d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["within-event"]
        return baker_jayaram(t1, t2, d1, d2, d3, d4, d5)


class DummyCorrelationModel(BaseAvgSACorrelationModel):
    """
    Dummy function returning just 1 (used as default function handle)
    """

    def build_correlation_matrix(self):
        self.rho = np.ones([len(self.avg_periods), len(self.avg_periods)])

    @staticmethod
    def get_correlation(t1, t2):
        """
        Computes the correlation coefficient for the specified periods.

        :param float t1:
            First period of interest.

        :param float t2:
            Second period of interest.

        :return float:
            The predicted correlation coefficient.
        """

        return 1.


class AristeidouCorrelationModel(BaseAvgSACorrelationModel):
    """
    Produce inter-period correlation for any two spectral periods.
    Subroutine taken from: https://usgs.github.io/shakemap/shakelib
    Based upon:
    Savvinos Aristeidou, Davit Shahnazaryan, and Gerard J. O'Reilly,
    "Correlation Models for Next-Generation Amplitude and
    Cumulative Intensity Measures using Artificial Neural Networks"
    (2024, Earthquake Spectra, Available at:
    https://doi.org/10.1177/87552930241270563).
    """

    def build_correlation_matrix(self):
        """
        Constucts the correlation matrix period-by-period from the
        correlation functions
        """
        self.rho = np.eye(len(self.avg_periods))
        for i, t1 in enumerate(self.avg_periods):
            for j, t2 in enumerate(self.avg_periods[i:]):
                self.rho[i, i + j] = self.get_correlation(t1, t2)
        self.rho += (self.rho.T - np.eye(len(self.avg_periods)))

    @staticmethod
    def get_correlation(t1, t2):
        """
        Computes the correlation coefficient for the specified periods.

        :param float t1:
            First period of interest.

        :param float t2:
            Second period of interest.

        :return float rho:
            The predicted correlation coefficient.
        """
        def read_json(filename: Union[Path, dict]):
            if isinstance(filename, Path) or isinstance(filename, str):
                filename = Path(filename)

                with open(filename) as f:
                    filename = json.load(f)

            return filename

        def linear(x):
            return x

        def tanh(x):
            return np.tanh(x)

        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def generate_function(x, biases, weights):
            biases = np.asarray(biases)
            weights = np.asarray(weights).T

            return biases.reshape(1, -1) + np.dot(weights, x.T).T

        ACTIVATION_FUNCTIONS = {
            "linear": linear,
            "softmax": softmax,
            "tanh": tanh,
            "sigmoid": sigmoid,
        }

        MODELS_ANN = read_json(
            Path(__file__).parents[2] / "assets/corr_ann.json")

        if t1 == t2:
            return 1.0

        model = MODELS_ANN["SA-SA"]
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        x = np.array([t_max, t_min])

        biases = model["biases"]
        weights = model["weights"]
        act_funcs = model["activation-functions"]

        for i, act_func in enumerate(act_funcs):
            activation = ACTIVATION_FUNCTIONS[act_func]

            _data = generate_function(x, biases[i], weights[i])
            x = activation(_data)

        return float(x)


CORRELATION_FUNCTION_HANDLES = {
    'baker_jayaram': BakerJayaramCorrelationModel,
    'akkar': AkkarCorrelationModel,
    'aristeidou': AristeidouCorrelationModel,
    'eshm20': ESHM20CorrelationModel,
    'none': DummyCorrelationModel
}
