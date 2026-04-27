# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

# flake8: noqa

import numpy as np

from ..coeffs_table import CoeffsTable
from ..base import GMPE
from .. import const
from ..imt import RSD575, IMT
from ..contexts import Context


def get_magnitude_term(C, mag):
    """
    Returns linear magnitude scaling term
    """
    return C["b0"] + C["b1"] * (mag - 6) + C["b2"] * (mag - 6) ** 2


def get_distance_term(C, rrup, mag):
    """
    Returns distance scaling term
    """
    fac = rrup > 100
    rmax100 = rrup.copy()
    rmax100[rmax100 > 100] = 100
    fr = C["b3"] * np.log(np.sqrt(
        rmax100 ** 2 + (np.exp(C["b4"] + C["b5"] * (mag - 6))) ** 2)) + \
        fac * (
        C["b6"] * np.log(np.sqrt(
            rrup ** 2 + (np.exp(C["b4"] + C["b5"] * (mag - 6))) ** 2))
        - C["b6"] * np.log(np.sqrt(
            100 ** 2 + (np.exp(C["b4"] + C["b5"] * (mag - 6))) ** 2))
    )
    return fr


def get_site_amplification(C, vs30):
    """
    Returns linear site amplification term
    """
    return C["b7"] * np.log(vs30 / 1000)


class VanHoutteEtAl2018RSD(GMPE):
    """
    Implements the GMPE of Van Houtte et al. (2018) for significant duration
    with 5 - 75 % Arias Intensity. doi:10.1785/0120170076. The oscillator
    duration model has not yet been implemented.
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are 5 - 75 % Arias
    #: significant duration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([RSD575])

    #: Supported intensity measure component is RotD50
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types are total, inter and intra-event
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Requires vs30
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameter is magnitude
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Required distance measure is closest distance to rupture
    REQUIRES_DISTANCES = {'rrup'}

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
        C = self.COEFFS[imt]
        mean = (get_magnitude_term(C, ctx.mag)
                   + get_distance_term(C, ctx.rrup, ctx.mag)
                   + get_site_amplification(C, ctx.vs30))
        sig = np.array([np.sqrt(C["tau"] ** 2. + C["phi"] ** 2.)])
        tau = np.array([C["tau"]])
        phi = np.array([C["phi"]])

        return mean, [sig, tau, phi]

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt          b0       b1      b2       b3      b4       b5       b6      b7     tau     phi
    rsd575  -1.7204   0.2272  0.0967   0.8870  2.7641   0.5777   1.1700 -0.1413  0.2270  0.4163
    """)
