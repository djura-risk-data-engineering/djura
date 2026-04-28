# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

"""
Module exports
:class:`ChiouYoungs2008SWISS01`,
:class:`ChiouYoungs2008SWISS06`,
:class:`ChiouYoungs2008SWISS04`.
"""

# flake8: noqa
import numpy as np

from .. import const
from ..imt import IMT
from ..contexts import Context
from .chiou_youngs_2008 import ChiouYoungs2008, _get_ln_y_ref
from .chiou_youngs_2008_swiss_coeffs import (
    COEFFS_FS_ROCK_SWISS01, COEFFS_FS_ROCK_SWISS06, COEFFS_FS_ROCK_SWISS04)
from .utils_swiss_gmpe import _apply_adjustments

def get_nl(C, ln_y_ref, exp1, exp2):
    # b and c coeffs from eq. 10
    b = C['phi2'] * (exp1 - exp2)
    c = C['phi4']

    y_ref = np.exp(ln_y_ref)
    # eq. 20
    NL = b * y_ref / (y_ref + c)
    return NL


def get_tau(C, ctx):
    # eq. 19 to calculate inter-event standard error
    mag_test = np.clip(ctx.mag - 5., 0., 2.)
    tau = C['tau1'] + (C['tau2'] - C['tau1']) / 2 * mag_test
    return tau


class ChiouYoungs2008SWISS01(ChiouYoungs2008):

    """
    This class extends :class:ChiouYoungs2008,
    adjusted to be used for the Swiss Hazard Model [2014].
    This GMPE is valid for a fixed value of vs30=620m/s

    1) kappa value
       K-adjustments corresponding to model 01 - as prepared by Ben Edwards
       K-value for PGA were not provided but infered from SA[0.01s]
       the model considers a fixed value of vs30==620 to match the
       reference vs30=1100m/s

    2) small-magnitude correction

    3) single station sigma - inter-event magnitude/distance adjustment

    Disclaimer: these equations are modified to be used for the
    Swiss Seismic Hazard Model [2014].
    The use of these models in other models
    is the soly responsability of the hazard modeler.

    Model implemented by laurentiu.danciu@gmail.com
    """
    #: Supported standard deviation type is total, inter-event and intra-event
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Vs30 value representing typical rock conditions in Switzerland.
    #: confirmed by the Swiss GMPE group
    DEFINED_FOR_REFERENCE_VELOCITY = 1105.

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
        ctx = ctx.copy()
        ctx.vs30 = 620 * np.ones(len(ctx.vs30))
        log_phi_ss = 1
        mean, [sig, tau, phi] = super().get_mean_and_stddevs(ctx, imt)

        t = get_tau(ChiouYoungs2008.COEFFS[imt], ctx)

        ln_y_ref = _get_ln_y_ref(ctx, ChiouYoungs2008.COEFFS[imt])

        exp1 = np.exp(ChiouYoungs2008.COEFFS[imt]['phi3'] *
                        (ctx.vs30.clip(-np.inf, 1130) - 360))

        exp2 = np.exp(ChiouYoungs2008.COEFFS[imt]['phi3'] * (1130 - 360))

        nl = get_nl(ChiouYoungs2008.COEFFS[imt], ln_y_ref, exp1, exp2)

        _apply_adjustments(
            ChiouYoungs2008.COEFFS, self.COEFFS_FS_ROCK[imt], 1,
            mean, sig, tau, phi, ctx, ctx.rjb, imt,
            log_phi_ss, NL=nl, tau_value=t)
        
        return mean, [sig, tau, phi]

    COEFFS_FS_ROCK = COEFFS_FS_ROCK_SWISS01


class ChiouYoungs2008SWISS06(ChiouYoungs2008SWISS01):
    """
    This class extends :class:ChiouYoungs2008,following same strategy
    as for :class:ChiouYoungs2008SWISS01 to be used for the
    Swiss Hazard Model [2014].
    """
    COEFFS_FS_ROCK = COEFFS_FS_ROCK_SWISS06


class ChiouYoungs2008SWISS04(ChiouYoungs2008SWISS01):
    """
    This class extends :class:ChiouYoungs2008,following same strategy
    as for :class:ChiouYoungs2008SWISS01 to be used for the
    Swiss Hazard Model [2014].
    """
    COEFFS_FS_ROCK = COEFFS_FS_ROCK_SWISS04
