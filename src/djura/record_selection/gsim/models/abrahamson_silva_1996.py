# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import numpy as np

from ..base import GMPE
from .. import const
from ..imt import RSD595, RSD575, IMT
from ..contexts import Context


class AbrahamsonSilva1996(GMPE):

    #: Supported intensity measure types
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {RSD595, RSD575}

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Requires sites parameters
    REQUIRES_SITES_PARAMETERS = {'soil'}

    #: Required rupture parameters
    REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    #: Required distance measures
    REQUIRES_DISTANCES = {'rrup'}

    def get_mean_and_stddevs(self, imt: IMT, ctx: Context):
        """
        Provides the ground motion prediction equation for Significant duration
        defined at the time from 5-75% or 5-95% of the Arias intensity

        Reference: Abrahamson, N. A. and Silva, W. J., 1996. Empirical
        ground motion models. Report to Brookhaven National Laboratory.

        Equations given in: Stewart, J. P., Chiou, S. J., Bray, J., Graves,
        R. W., Somerville, P. G. and Abrahamson, N. A., 2001. Ground motion
        evaluation procedures for performance-based design. PEER 2001/09,
        Pacific Earthquake Engineering Research Centre.

        Parameters
        ----------
        ctx : Context
            Instance of Context class which contains the site parameters and,
            rupture and site-to-source distance parameters of the scenario.
        imt : IMT
            Instance of IMT class which describes the intensity measure type.

        Returns
        -------
        Tuple containing
            mean SD
            lognormal standard deviation in SD
        (both in exponential form)
        """

        imt = str(imt)

        if imt == "RSD575":
            imt = "Ds575"
        if imt == "RSD595":
            imt = "Ds595"

        if imt == 'Ds575':
            # Coefficients for Ds5-75
            beta = 3.2
            b1 = 5.204
            b2 = 0.851
            mstar = 6
            c1 = 0.805
            c2 = 0.063
            rc = 10
            d_rat = 0
            sigma = 0.55

        elif imt == 'Ds595':
            # Coefficients for Ds5-95
            beta = 3.2
            b1 = 5.204
            b2 = 0.851
            mstar = 6
            c1 = 0.805
            c2 = 0.063
            rc = 10
            d_rat = 0.845
            sigma = 0.49

        if ctx.rrup >= rc:
            sd = np.exp(
                np.log(
                    (
                        (np.exp(b1 + b2 * (ctx.mag - mstar)))
                        / (10 ** (1.5 * ctx.mag + 16.05))
                    )
                    ** (-1 / 3)
                    / (4.9 * 10**6 * beta)
                    + ctx.soil * c1
                    + c2 * (ctx.rrup - rc)
                )
                + d_rat
            )
        else:
            sd = np.exp(
                np.log(
                    (
                        np.exp(b1 + b2 * (ctx.mag - mstar))
                        / (10 ** (1.5 * ctx.mag + 16.05))
                    )
                    ** (-1 / 3)
                    / (4.9 * 10**6 * beta)
                    + ctx.soil * c1
                )
                + d_rat
            )

        median_sd = sd
        sigma_lnsd = sigma

        return np.log(median_sd), np.array([sigma_lnsd])
