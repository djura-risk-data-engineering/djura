# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

# flake8: noqa

import numpy as np

from ..coeffs_table import CoeffsTable
from ..base import GMPE
from .. import const
from ..imt import RSD595, RSD575, IMT
from ..contexts import Context


def get_magnitude_term(C, mag):
    """
    Returns linear magnitude scaling term
    """
    return C["c0"] + C["m1"] * mag


def get_distance_term(C, rrup, mag):
    """
    Returns distance scaling term
    """
    return (C["r1"] + C["r2"] * mag) *\
        np.log(np.sqrt(rrup ** 2. + C["h1"] ** 2.))


def get_ztor_term(C, ztor):
    """
    Returns depth to top of rupture scaling
    """
    return C["z1"] * ztor


def get_site_amplification(C, vs30):
    """
    Returns linear site amplification term
    """
    return C["v1"] * np.log(vs30)


def get_stddevs(C):
    """
    Returns the standard deviations
    """
    return [np.sqrt(C["tau"] ** 2. + C["phi"] ** 2.), C["tau"], C["phi"]]


class BommerEtAl2009RSD(GMPE):
    """
    Implements the GMPE of Bommer et al. (2009) for significant duration with
    5 - 75 % Arias Intensity and 5 - 95 % Arias Intensity
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are 5 - 95 % Arias and 5 - 75 % Arias
    #: significant duration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {RSD595, RSD575}

    #: Supported intensity measure component is the geometric mean horizontal
    #: component
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation type is only total, see table 7, page 35
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Requires vs30
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are magnitude and top of rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

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
                + get_ztor_term(C, ctx.ztor)
                + get_site_amplification(C, ctx.vs30))

        sig, tau, phi = get_stddevs(C)

        return mean, [
            np.array([sig]), 
            np.array([tau]),
            np.array([phi])
        ]

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt          c0       m1      r1       r2      h1       v1       z1     tau     phi
    rsd575  -5.6298   1.2619  2.0063  -0.2520  2.3316  -0.2900  -0.0522  0.3527  0.4304
    rsd595  -2.2393   0.9368  1.5686  -0.1953  2.5000  -0.3478  -0.0365  0.3252  0.3460
    """)
