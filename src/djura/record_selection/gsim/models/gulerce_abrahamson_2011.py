# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

# flake8: noqa

"""
Module exports :class:`GulerceAbrahamson2011`
"""

import copy
import numpy as np

from .. import const
from ..contexts import Context
from ..coeffs_table import CoeffsTable
from ..base import GMPE
from ..imt import PGA, PGV, SA, IMT
from .abrahamson_silva_2008 import AbrahamsonSilva2008


#: equation constants (that are IMT independent); see "fixed inputs" in the
#: MATLAB reference and Table 2, page 1028.
CONSTS = {
    'c1': 6.75,
    'c4': 10.,
    'a3': 0.0147,
    'a4': 0.0334,
    'a5': -0.034,
    'n': 1.18,
    'c': 1.88}


def _compute_base_term(C, ctx):
    """
    Compute and return base model term, that is the first term, f1, in
    Equation 6, page 1028. The calculation of this term is described in
    Equation 1, page 1027.
    """
    c1 = CONSTS['c1']
    R = np.sqrt(ctx.rrup ** 2 + CONSTS['c4'] ** 2)

    base_term = (C['a1'] +
                 C['a8'] * ((8.5 - ctx.mag) ** 2) +
                 (C['a2'] + CONSTS['a3'] * (ctx.mag - c1)) *
                 np.log(R))
    extra = np.where(ctx.mag <= c1,
                     CONSTS['a4'] * (ctx.mag - c1),
                     CONSTS['a5'] * (ctx.mag - c1))
    return base_term + extra


def _compute_faulting_style_term(C, ctx):
    """
    Compute and return faulting style term, that is the sum of the second
    and third terms in Equation 6, page 1028.
    """
    # ranges of rake values for each faulting mechanism are specified in
    # section "Functional Form of the Model", page 1029. In the standalone
    # MATLAB these are passed in directly as the FRV and FNM flags.
    frv = (30 < ctx.rake) & (ctx.rake < 150)
    fnm = (-120 < ctx.rake) & (ctx.rake < -60)
    return C['a6'] * frv + C['a7'] * fnm


def _compute_site_response_term(C, imt, ctx, pga1100):
    """
    Compute and return site response model term, that is the fourth term
    f5 in Equation 6, page 1028. Note the change in sign for the piecewise
    term from the adopted equation in AS08.
    """
    site_resp_term = np.zeros_like(ctx.vs30)

    vs30_star, _ = _compute_vs30_star_factor(imt, ctx.vs30)
    vlin, c, n = C['VLIN'], CONSTS['c'], CONSTS['n']
    a10, b = C['a10'], C['b']

    idx = vs30_star < vlin
    arg = vs30_star[idx] / vlin
    site_resp_term[idx] = (a10 * np.log(arg) +
                           b * np.log(pga1100[idx] + c) -
                           b * np.log(pga1100[idx] + c * (arg ** n)))

    idx = ~idx
    site_resp_term[idx] = (a10 - b * n) * np.log(vs30_star[idx] / vlin)

    return site_resp_term


def _get_stddevs(C, ctx):
    """
    Return standard deviations as described in Equations 7 to 9, page 1029.
    The aleatory variability depends on magnitude only.
    """
    std_intra = _compute_intra_event_std(C, ctx.mag)
    std_inter = _compute_inter_event_std(C, ctx.mag)
    return [np.sqrt(std_intra ** 2 + std_inter ** 2), std_inter, std_intra]


def _compute_intra_event_std(C, mag):
    """
    Intra-event (within-event) standard deviation, Equation 7, page 1029.
    """
    sigma_0 = C['s1'] + (C['s2'] - C['s1']) * (mag - 5) / 2
    sigma_0[mag < 5] = C['s1']
    sigma_0[mag > 7] = C['s2']
    return sigma_0


def _compute_inter_event_std(C, mag):
    """
    Inter-event (between-event) standard deviation, Equation 8, page 1029.
    """
    tau_0 = C['s3'] + (C['s4'] - C['s3']) * (mag - 5) / 2
    tau_0[mag < 5] = C['s3']
    tau_0[mag > 7] = C['s4']
    return tau_0


def _compute_vs30_star_factor(imt, vs30):
    """
    Compute and return vs30 star factor, Equation 4, page 1028.
    """
    v1 = _compute_v1_factor(imt)
    vs30_star = vs30.copy()
    vs30_star[vs30_star >= v1] = v1

    return vs30_star, v1


def _compute_v1_factor(imt):
    """
    Compute and return v1 factor, Equation 5, page 1028.
    """
    if imt.string[:2] == "SA":
        t = imt.period
        if t <= 0.50:
            v1 = 1500.0
        elif 0.5 < t <= 1.0:
            v1 = np.exp(8.0 - 0.795 * np.log(t / 0.21))
        elif 1.0 < t < 2.0:
            v1 = np.exp(6.76 - 0.297 * np.log(t))
        else:
            v1 = 700.0
    elif imt.string == "PGA":
        v1 = 1500.0
    else:
        # this is for PGV
        v1 = 862.0

    return v1


class GulerceAbrahamson2011(GMPE):
    """
    Implements the GMPE by Gulerce & Abrahamson (2011) for the
    vertical-to-horizontal (V/H) ratio model derived using ground motions from
    the PEER NGA-West1 Project.

    This model follows the same functional form as in AS08 by Abrahamson &
    Silva (2008) with minor modifications to the underlying parameters. The
    median V/H ratio requires the horizontal rock (Vs30 = 1100 m/s) PGA, which
    is obtained from :class:`AbrahamsonSilva2008`, the horizontal GMPE the
    model was developed with (``gmpe_as_2008`` in the MATLAB reference).

    Unlike the standalone MATLAB routine - which only receives
    ``(M, Rrup, Vs30, FRV, FNM)`` and therefore assumes a vertical strike-slip
    geometry when calling AS08 - here the full rupture context is passed
    through to AS08, consistent with the other vertical GMPEs in this package
    (e.g. :class:`GulerceEtAl2017`). The aleatory variability depends on
    magnitude only and is identical in both approaches.

    Reference:

    Gulerce, Z. & Abrahamson, N. (2011), "Site-Specific Design Spectra for
    Vertical Ground Motion", Earthquake Spectra, 27(4), 1023-1047.

    Adapted from the MATLAB implementation by N. Simon Kwong.
    """
    #: Supported tectonic region type is active shallow crust, as part of the
    #: NGA-West1 Database; re-defined here for clarity.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are peak ground acceleration, peak
    #: ground velocity, and spectral acceleration at T=0.01 to 10.0 s; see
    #: Table 3, page 1030
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is the
    #: :attr:`~openquake.hazardlib.const.IMC.VERTICAL_TO_HORIZONTAL_RATIO`
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = (
        const.IMC.VERTICAL_TO_HORIZONTAL_RATIO)

    #: Supported standard deviation types are inter-event, intra-event
    #: and total; see Equations 7 to 9, page 1029.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Required site parameter is Vs30 only. Unlike in AS08, the nonlinear
    #: site response and Z1.0 scaling is not available for the vertical
    #: component; see section for "Functional Form of the Model", Equation 3.
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are magnitude and rake; see section for
    #: "Functional Form of the Model", Equation 6.
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake'}

    #: Required distance measure is Rrup; see the section for "Functional
    #: Form of the Model", Equation 1.
    REQUIRES_DISTANCES = {'rrup'}

    #: Verification of mean value data was done by digitizing Figures 7 to 10
    #: using https://apps.automeris.io/wpd/ . Only SA is covered.
    non_verified = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The reference horizontal GMPE used to obtain the rock (Vs30 = 1100)
        # PGA, as in the MATLAB reference (gmpe_as_2008).
        self.gmpe = AbrahamsonSilva2008()

        # The requirements of the reference horizontal GMPE must also be met,
        # since it is evaluated to obtain the rock PGA.
        self.REQUIRES_SITES_PARAMETERS = (
            self.REQUIRES_SITES_PARAMETERS
            | self.gmpe.REQUIRES_SITES_PARAMETERS)
        self.REQUIRES_RUPTURE_PARAMETERS = (
            self.REQUIRES_RUPTURE_PARAMETERS
            | self.gmpe.REQUIRES_RUPTURE_PARAMETERS)
        self.REQUIRES_DISTANCES = (
            self.REQUIRES_DISTANCES | self.gmpe.REQUIRES_DISTANCES)

    def get_mean_and_stddevs(self, ctx: Context, imt: IMT):
        """
        Gets the logarithmic mean V/H ratio and its standard deviations.

        Parameters
        ----------
        ctx : Context
            Instance of Context class which contains the site parameters and,
            rupture and site-to-source distance parameters of the scenario.
        imt : IMT
            Instance of IMT class which describes the intensity measure type.

        Returns
        -------
        numpy.ndarray and list of numpy.ndarray
            Mean log V/H ratio and [total, inter-event, intra-event] sigmas.
        """
        # Compute median PGA on rock (Vs30 = 1100 m/s) from the reference
        # horizontal GMPE; needed for the nonlinear site response term.
        ctx_rock = copy.copy(ctx)
        ctx_rock.vs30 = np.full_like(ctx.vs30, 1100.)
        mean_pga, _ = self.gmpe.get_mean_and_stddevs(ctx_rock, PGA())
        pga1100 = np.exp(np.asarray(mean_pga)).reshape(-1)

        C = self.COEFFS[imt]
        mean = (_compute_base_term(C, ctx) +
                _compute_faulting_style_term(C, ctx) +
                _compute_site_response_term(C, imt, ctx, pga1100))
        sig, tau, phi = _get_stddevs(C, ctx)

        return mean, [sig, tau, phi]

    #: Coefficients obtained from Table 3, page 1030
    #: Note the atypical periods 0.029 s and 0.260 s used.
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT    VLIN     b       a1      a2      a6      a7      a8      a10     s1      s2      s3      s4
    pga     865.1  -1.186   0.140  -0.160  -0.105   0.000   0.003  -1.230   0.422   0.333   0.213   0.161
    pgv     400.0  -1.955  -1.200   0.090   0.050   0.150   0.022  -1.847   0.373   0.369   0.234   0.170
    0.010   865.1  -1.186   0.140  -0.160  -0.105   0.000   0.003  -1.230   0.450   0.330   0.230   0.150
    0.020   865.1  -1.219   0.140  -0.160  -0.105   0.000   0.003  -1.268   0.450   0.330   0.230   0.150
    0.029   898.6  -1.269   0.335  -0.185  -0.140   0.000   0.003  -1.366   0.450   0.330   0.230   0.150
    0.040   994.5  -1.308   0.562  -0.238  -0.160   0.000   0.003  -1.457   0.450   0.341   0.230   0.150
    0.050  1053.5  -1.346   0.720  -0.275  -0.136   0.000  -0.001  -1.533   0.450   0.351   0.230   0.150
    0.075  1085.7  -1.471   0.552  -0.240  -0.019   0.000  -0.007  -1.706   0.450   0.370   0.230   0.150
    0.100  1032.5  -1.624   0.214  -0.169   0.000   0.017  -0.010  -1.831   0.450   0.384   0.230   0.150
    0.150   877.6  -1.931  -0.262  -0.069   0.000   0.040  -0.008  -2.114   0.450   0.403   0.230   0.150
    0.200   748.2  -2.188  -0.600   0.002   0.000   0.057  -0.003  -2.362   0.450   0.416   0.230   0.150
    0.260   639.0  -2.412  -0.769   0.023   0.000   0.072   0.001  -2.527   0.450   0.429   0.230   0.150
    0.300   587.1  -2.518  -0.861   0.034   0.000   0.080   0.006  -2.598   0.450   0.436   0.230   0.150
    0.400   503.0  -2.657  -1.045   0.057   0.000   0.097   0.015  -2.685   0.450   0.449   0.230   0.150
    0.500   456.6  -2.669  -1.189   0.075   0.000   0.110   0.022  -2.657   0.450   0.460   0.230   0.150
    0.750   410.5  -2.401  -1.250   0.090   0.000   0.133   0.022  -2.265   0.450   0.479   0.237   0.150
    1.000   400.0  -1.955  -1.209   0.090   0.000   0.150   0.022  -1.685   0.450   0.492   0.266   0.150
    1.500   400.0  -1.025  -1.152   0.090   0.029   0.150   0.022  -0.570   0.450   0.511   0.307   0.150
    2.000   400.0  -0.299  -1.111   0.090   0.050   0.150   0.022   0.250   0.532   0.520   0.337   0.150
    3.000   400.0   0.000  -1.054   0.090   0.079   0.150   0.022   0.460   0.648   0.520   0.378   0.213
    4.000   400.0   0.000  -1.014   0.090   0.100   0.150   0.022   0.460   0.700   0.520   0.407   0.258
    5.000   400.0   0.000  -1.000   0.090   0.100   0.150   0.022   0.460   0.700   0.520   0.430   0.292
    7.500   400.0   0.000  -1.000   0.090   0.100   0.150   0.022   0.460   0.700   0.520   0.471   0.355
    10.00   400.0   0.000  -1.000   0.090   0.100   0.150   0.022   0.460   0.700   0.520   0.500   0.400
    """)
