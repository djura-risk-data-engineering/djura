# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

"""
Module exports :class:`AkkarCagnan2010`.
"""

# flake8: noqa
import numpy as np
from scipy.constants import g

from .atkinson_boore_2006 import (
    _get_site_amplification_linear,
    _get_site_amplification_non_linear,
)
from .boore_atkinson_2008 import BooreAtkinson2008
from ..coeffs_table import CoeffsTable
from ..base import GMPE
from .. import const
from ..imt import PGA, PGV, SA, IMT
from ..contexts import Context

# c1 is the reference magnitude, fixed to 6.5
# see paragraph 'Functional Form', p. 2982
c1 = 6.5


def _compute_faulting_style_term(C, rake):
    """
    Compute and return fifth and sixth terms in equations (1a)
    and (1b), pages 2981 and 2982, respectively.
    """
    Fn = (rake > -135.0) & (rake < -45.0)
    Fr = (rake > 45.0) & (rake < 135.0)
    return C["a8"] * Fn + C["a9"] * Fr


def _compute_linear_magnitude_term(C, mag):
    """
    Compute and return second term in equations (1a)
    and (1b), pages 2981 and 2982, respectively.
    """
    return np.where(mag <= c1, C["a2"] * (mag - c1), C["a3"] * (mag - c1))


def _compute_logarithmic_distance_term(C, mag, rjb):
    """
    Compute and return fourth term in equations (1a)
    and (1b), pages 2981 and 2982, respectively.
    """
    return (C["a5"] + C["a6"] * (mag - c1)) * np.log(np.sqrt(rjb**2 + C["a7"] ** 2))


def _compute_mean(C, mag, rjb, rake):
    """
    Compute and return mean value without site conditions,
    that is equations (1a) and (1b), p.2981-2982.
    """
    mean = (
        C["a1"]
        + _compute_linear_magnitude_term(C, mag)
        + _compute_quadratic_magnitude_term(C, mag)
        + _compute_logarithmic_distance_term(C, mag, rjb)
        + _compute_faulting_style_term(C, rake)
    )

    return mean


def _compute_quadratic_magnitude_term(C, mag):
    """
    Compute and return third term in equations (1a)
    and (1b), pages 2981 and 2982, respectively.
    """
    return C["a4"] * (8.5 - mag) ** 2


def _get_stddevs(C):
    """
    Return standard deviations as defined in table 3, p. 2985.
    """
    return [np.sqrt(C["sigma"] ** 2 + C["tau"] ** 2), C["tau"], C["sigma"]]


class AkkarCagnan2010(GMPE):
    """
    Implements GMPE developed by Sinnan Akkar and Zehra Cagnan and
    published as "A Local Ground-Motion Predictive Model for Turkey,
    and Its Comparison with Other Regional and Global Ground-Motion
    Models" (2010, Bulletin of the Seismological Society of America,
    Volume 100, No. 6, pages 2978-2995). It uses the same site response
    function used in Boore and Atkinson 2008.
    """

    #: Supported tectonic region type is active shallow crust (the
    #: equations being developed for Turkey, see paragraph 'Strong Motion
    #: Databank', p. 2981)
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration, see paragraph
    # 'Functional Form', p. 2981
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is geometric mean
    #: of two horizontal components :
    #: attr:`~openquake.hazardlib.const.IMC.GEOMETRIC_MEAN`, see paragraph
    #: 'Functional Form', p. 2981.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see Table 3, p. 2985.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT,
    }

    #: Required site parameters is Vs30.
    #: See paragraph 'Functionl Form', p. 2981.
    REQUIRES_SITES_PARAMETERS = {"vs30"}

    #: Required rupture parameters are magnitude, and rake.
    #: See paragraph 'Functional Form', p. 2981.
    REQUIRES_RUPTURE_PARAMETERS = {"mag", "rake"}

    #: Required distance measure is Rjb.
    #: See paragraph 'Functional Form', p. 2981.
    REQUIRES_DISTANCES = {"rjb"}

    def get_mean_and_stddevs(self, ctx: Context, imt: IMT):
        """
        Gets the logarithmic mean and standard deviations.

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
        """
        # compute median PGA on rock, needed to compute non-linear site
        # amplification
        C = self.COEFFS_AC10[PGA()]
        pga4nl = np.exp(_compute_mean(C, ctx.mag, ctx.rjb, ctx.rake)) * 1e-2 / g

        C = self.COEFFS_AC10[imt]
        C_SR = self.COEFFS_SOIL_RESPONSE[imt]

        # compute full mean value by adding site amplification terms
        # (but avoiding recomputing mean on rock for PGA)
        if imt.string == "PGA":
            mean = (
                np.log(pga4nl)
                + _get_site_amplification_linear(ctx.vs30, C_SR)
                + _get_site_amplification_non_linear(ctx.vs30, pga4nl, C_SR)
            )
        else:
            mean = (
                _compute_mean(C, ctx.mag, ctx.rjb, ctx.rake)
                + _get_site_amplification_linear(ctx.vs30, C_SR)
                + _get_site_amplification_non_linear(ctx.vs30, pga4nl, C_SR)
            )

        # convert from cm/s**2 to g for SA (PGA is already computed in g)
        if imt.string[:2] == "SA":
            mean = np.log(np.exp(mean) * 1e-2 / g)

        sig, tau, phi = _get_stddevs(C)

        return mean, [np.array([sig]), np.array([tau]), np.array([phi])]

    #: Coefficient table (from Table 3, p. 2985)
    #: sigma is the 'intra-event' standard deviation,
    #: while tau is the 'inter-event' standard deviation
    COEFFS_AC10 = CoeffsTable(
        sa_damping=5,
        table="""\
    IMT   a1        a2     a3     a4       a5       a6       a7       a8       a9       sigma   tau
    pgv   5.60931  -0.513 -0.695 -0.25800 -0.90393  0.21576  5.57472 -0.10481  0.07791  0.6154  0.526
    pga   8.92418  -0.513 -0.695 -0.18555 -1.25594  0.18105  7.33617 -0.02125  0.01851  0.6527  0.5163
    0.03  8.85984  -0.513 -0.695 -0.17123 -1.25132  0.18421  7.46968 -0.0134   0.03512  0.6484  0.5148
    0.05  9.05262  -0.513 -0.695 -0.15516 -1.28796  0.1984   7.26552  0.02076  0.01484  0.6622  0.5049
    0.075 9.56670  -0.513 -0.695 -0.13840 -1.38817  0.20246  8.03646  0.07311  0.02492  0.6849  0.5144
    0.10  9.85606  -0.513 -0.695 -0.11563 -1.43846  0.21833  8.84202  0.11044 -0.00620  0.7001  0.5182
    0.15  10.43715 -0.513 -0.695 -0.17897 -1.46786  0.15588  9.39515  0.03555  0.19751  0.6958  0.549
    0.20  10.63516 -0.513 -0.695 -0.21034 -1.44625  0.11590  9.60868 -0.03536  0.18594  0.6963  0.5562
    0.25  10.12551 -0.513 -0.695 -0.25565 -1.27388  0.09426  7.54353 -0.10685  0.13574  0.7060  0.5585
    0.30  10.12745 -0.513 -0.695 -0.27020 -1.26899  0.08352  8.03144 -0.10685  0.13574  0.6718  0.5735
    0.40  9.47855  -0.513 -0.695 -0.30498 -1.09793  0.06082  6.24042 -0.11197  0.16555  0.6699  0.5857
    0.50  8.95147  -0.513 -0.695 -0.29877 -1.01703  0.09099  5.67936 -0.10118  0.23546  0.6455  0.5782
    0.75  8.10498  -0.513 -0.695 -0.3349  -0.84365  0.08647  4.93842 -0.0456   0.10993  0.6463  0.6168
    1.00  7.61737  -0.513 -0.695 -0.35366 -0.75840  0.09623  4.12590 -0.01936  0.19729  0.6485  0.6407
    1.50  7.20427  -0.513 -0.695 -0.39858 -0.70134  0.11219  3.46535 -0.02618  0.21977  0.6300  0.6751
    2.00  6.70845  -0.513 -0.695 -0.39528 -0.70766  0.12032  3.8822  -0.03215  0.20584  0.6243  0.6574
    """,
    )

    COEFFS_SOIL_RESPONSE = BooreAtkinson2008.COEFFS_SOIL_RESPONSE
