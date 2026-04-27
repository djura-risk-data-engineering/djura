# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

"""
Module exports :class:`NGAEastGMPE` and :class:`NGAEastGMPETotalSigma`
"""
import numpy as np
from ..coeffs_table import CoeffsTable
from .. import const
from ..imt import PGA, SA
from .gmpe_table import GMPETable, _get_mean


def _scaling(mean_tau, sd_tau2):
    """
    Returns the chi-2 scaling factor from the mean and variance of the
    uncertainty model, as reported in equation 5.4 of Al Atik (2015)
    """
    return sd_tau2 ** 2. / (2.0 * mean_tau ** 2.)


def _dof(mean_tau, sd_tau2):
    """
    Returns the degrees of freedom for the chi-2 distribution from the mean and
    variance of the uncertainty model, as reported in equation 5.5 of Al Atik
    (2015)
    """
    return 2.0 * mean_tau ** 4. / sd_tau2 ** 2.


def _at_percentile(tau, var_tau, percentile):
    """
    Returns the value of the inverse chi-2 distribution at the given
    percentile from the mean and variance of the uncertainty model, as
    reported in equations 5.1 - 5.3 of Al Atik (2015)
    """
    from scipy.stats import chi2

    assert percentile >= 0.0 and percentile <= 1.0
    c_val = _scaling(tau, var_tau)
    k_val = _dof(tau, var_tau)
    return np.sqrt(c_val * chi2.ppf(percentile, df=k_val))


def get_tau_at_quantile(mean, stddev, quantile):
    """
    Returns the value of tau at a given quantile in the form of a dictionary
    organised by intensity measure
    """
    tau_model = {}
    for imt in mean:
        tau_model[imt] = {}
        for key in mean[imt]:
            if quantile is None:
                tau_model[imt][key] = mean[imt][key]
            else:
                tau_model[imt][key] = _at_percentile(mean[imt][key],
                                                     stddev[imt][key],
                                                     quantile)
    return tau_model


def ITPL(mag, tu, tl, ml, f):
    return tl + (tu - tl) * (mag - ml) / f


# Mean tau values from the CENA tau model
CENA_TAU_MEAN = {
    "PGV": {"tau1": 0.3477, "tau2": 0.3281, "tau3": 0.3092},
    "SA": {"tau1": 0.3730, "tau2": 0.3375, "tau3": 0.3064}
}

# Standard deviation of tau values from the CENA tau model
CENA_TAU_SD = {
    "PGV": {"tau1": 0.0554, "tau2": 0.0477, "tau3": 0.0449},
    "SA": {"tau1": 0.0688, "tau2": 0.0661, "tau3": 0.0491}
}

# Mean tau values from the CENA constant-tau model
CENA_CONSTANT_TAU_MEAN = {"PGV": {"tau": 0.3441}, "SA": {"tau": 0.3695}}

# Standard deviation of tau values from CENA constant-tau model
CENA_CONSTANT_TAU_SD = {"PGV": {"tau": 0.0554}, "SA": {"tau": 0.0688}}

# Mean tau values from the global model - Table 5.1
GLOBAL_TAU_MEAN = {
    "PGV": {"tau1": 0.3733, "tau2": 0.3639, "tau3": 0.3434, "tau4": 0.3236},
    "SA": {"tau1": 0.4518, "tau2": 0.4270, "tau3": 0.3863, "tau4": 0.3508}
}

# Standard deviation of tau values from the global model - Table 5.1
GLOBAL_TAU_SD = {
    "PGV": {"tau1": 0.0558, "tau2": 0.0554, "tau3": 0.0477, "tau4": 0.0449},
    "SA": {"tau1": 0.0671, "tau2": 0.0688, "tau3": 0.0661, "tau4": 0.0491}
}

TAU_SETUP = {
    "cena": {"MEAN": CENA_TAU_MEAN, "STD": CENA_TAU_SD},
    "cena_constant": {"MEAN": CENA_CONSTANT_TAU_MEAN,
                      "STD": CENA_CONSTANT_TAU_SD},
    "global": {"MEAN": GLOBAL_TAU_MEAN, "STD": GLOBAL_TAU_SD}
}


def cena_tau(imt, mag, params):
    """
    Returns the inter-event standard deviation, tau, for the CENA case
    """
    if imt.string == "PGV":
        C = params["PGV"]
    else:
        C = params["SA"]
    tau = np.full_like(mag, C["tau1"])
    tau[mag > 6.5] = C["tau3"]
    idx = (mag > 5.5) & (mag <= 6.5)
    tau[idx] = ITPL(mag[idx], C["tau3"], C["tau2"], 5.5, 1.0)
    idx = (mag > 5.0) & (mag <= 5.5)
    tau[idx] = ITPL(mag[idx], C["tau2"], C["tau1"], 5.0, 0.5)
    return tau


def cena_constant_tau(imt, mag, params):
    """
    Returns the inter-event tau for the constant tau case
    """
    if imt.string == "PGV":
        return params["PGV"]["tau"]
    else:
        return params["SA"]["tau"]


def global_tau(imt, mag, params):
    """
    'Global' model of inter-event variability, as presented in equation 5.6
    (p103)
    """
    if imt.string == "PGV":
        C = params["PGV"]
    else:
        C = params["SA"]
    tau = np.full_like(mag, C["tau1"])
    tau[mag > 6.5] = C["tau4"]
    idx = (mag > 5.5) & (mag <= 6.5)
    tau[idx] = ITPL(mag[idx], C["tau4"], C["tau3"], 5.5, 1.0)
    idx = (mag > 5.0) & (mag <= 5.5)
    tau[idx] = ITPL(mag[idx], C["tau3"], C["tau2"], 5.0, 0.5)
    idx = (mag > 4.5) & (mag <= 5.0)
    tau[idx] = ITPL(mag[idx], C["tau2"], C["tau1"], 4.5, 0.5)
    return tau


# Gather tau model implementation functions into dictionary
TAU_EXECUTION = {
    "cena": cena_tau,
    "cena_constant": cena_constant_tau,
    "global": global_tau}


def get_phi_ss_at_quantile(phi_model, quantile):
    """
    Returns the phi_ss values at the specified quantile as an instance of
    `class`:: openquake.hazardlib.gsim.base.CoeffsTable - applies to the
    magnitude-dependent cases
    """
    from copy import deepcopy

    # Setup SA coeffs - the backward compatible Python 2.7 way
    coeffs = deepcopy(phi_model.sa_coeffs)
    coeffs.update(phi_model.non_sa_coeffs)
    for imt in coeffs:
        if quantile is None:
            coeffs[imt] = {"a": phi_model[imt]["mean_a"],
                           "b": phi_model[imt]["mean_b"]}
        else:
            coeffs[imt] = {
                "a": _at_percentile(phi_model[imt]["mean_a"],
                                    phi_model[imt]["var_a"],
                                    quantile),
                "b": _at_percentile(phi_model[imt]["mean_b"],
                                    phi_model[imt]["var_b"],
                                    quantile)}
    return CoeffsTable.fromdict(coeffs)


# Phi_ss coefficients for the global model
PHI_SS_GLOBAL = CoeffsTable(sa_damping=5., table="""\
imt     mean_a   var_a  mean_b  var_b
pgv     0.5034  0.0609  0.3585  0.0316
pga     0.5477  0.0731  0.3505  0.0412
0.010   0.5477  0.0731  0.3505  0.0412
0.020   0.5464  0.0727  0.3505  0.0416
0.030   0.5450  0.0723  0.3505  0.0419
0.040   0.5436  0.0720  0.3505  0.0422
0.050   0.5424  0.0716  0.3505  0.0425
0.075   0.5392  0.0707  0.3505  0.0432
0.100   0.5361  0.0699  0.3505  0.0439
0.150   0.5299  0.0682  0.3543  0.0453
0.200   0.5240  0.0666  0.3659  0.0465
0.250   0.5183  0.0651  0.3765  0.0476
0.300   0.5127  0.0637  0.3876  0.0486
0.400   0.5022  0.0611  0.4066  0.0503
0.500   0.4923  0.0586  0.4170  0.0515
0.750   0.4704  0.0535  0.4277  0.0526
1.000   0.4519  0.0495  0.4257  0.0508
1.500   0.4231  0.0439  0.4142  0.0433
2.000   0.4026  0.0405  0.4026  0.0396
3.000   0.3775  0.0371  0.3775  0.0366
4.000   0.3648  0.0358  0.3648  0.0358
5.000   0.3583  0.0353  0.3583  0.0356
7.500   0.3529  0.0350  0.3529  0.0355
10.00   0.3519  0.0350  0.3519  0.0355
""")


# Phi_ss coefficients for the CENA model
PHI_SS_CENA = CoeffsTable(sa_damping=5., table="""\
imt     mean_a   var_a  mean_b   var_b
pgv     0.5636  0.0807  0.4013  0.0468
pga     0.5192  0.0693  0.3323  0.0364
0.010   0.5192  0.0693  0.3323  0.0364
0.020   0.5192  0.0693  0.3331  0.0365
0.030   0.5192  0.0693  0.3339  0.0365
0.040   0.5192  0.0693  0.3348  0.0367
0.050   0.5192  0.0693  0.3355  0.0367
0.075   0.5192  0.0693  0.3375  0.0370
0.100   0.5192  0.0693  0.3395  0.0372
0.150   0.5192  0.0693  0.3471  0.0382
0.200   0.5192  0.0693  0.3625  0.0402
0.250   0.5192  0.0693  0.3772  0.0423
0.300   0.5192  0.0693  0.3925  0.0446
0.400   0.5192  0.0693  0.4204  0.0492
0.500   0.5192  0.0693  0.4398  0.0527
0.750   0.5192  0.0693  0.4721  0.0590
1.000   0.5192  0.0693  0.4892  0.0626
1.500   0.5192  0.0693  0.5082  0.0668
2.000   0.5192  0.0693  0.5192  0.0693
3.000   0.5192  0.0693  0.5192  0.0693
4.000   0.5192  0.0693  0.5192  0.0693
5.000   0.5192  0.0693  0.5192  0.0693
7.500   0.5192  0.0693  0.5192  0.0693
10.00   0.5192  0.0693  0.5192  0.0693
""")


# Phi_ss coefficients for the CENA constant-phi model
PHI_SS_CENA_CONSTANT = CoeffsTable(sa_damping=5., table="""\
imt     mean_a    var_a   mean_b   var_b
pgv     0.5507   0.0678   0.5507  0.0678
pga     0.5132   0.0675   0.5132  0.0675
0.010   0.5132   0.0675   0.5132  0.0675
10.00   0.5132   0.0675   0.5132  0.0675
""")


# Gather the models to setup the phi_ss values for the given quantile
PHI_SETUP = {
    "cena": PHI_SS_CENA,
    "cena_constant": PHI_SS_CENA_CONSTANT,
    "global": PHI_SS_GLOBAL}

# Phi_s2ss coefficients for the CENA
PHI_S2SS_CENA = CoeffsTable(sa_damping=5., table="""\
imt       mean      var
pgv     0.4344   0.0200
pga     0.4608   0.0238
0.010   0.4608   0.0238
0.020   0.4617   0.0238
0.030   0.4700   0.0240
0.040   0.4871   0.0260
0.050   0.5250   0.0290
0.075   0.5800   0.0335
0.100   0.5930   0.0350
0.150   0.5714   0.0325
0.200   0.5368   0.0296
0.250   0.5058   0.0272
0.300   0.4805   0.0250
0.400   0.4440   0.0212
0.500   0.4197   0.0182
0.750   0.3849   0.0139
1.000   0.3667   0.0135
1.500   0.3481   0.0157
2.000   0.3387   0.0173
3.000   0.3292   0.0195
4.000   0.3245   0.0211
5.000   0.3216   0.0224
7.500   0.3178   0.0240
10.00   0.3159   0.0240
""")

# Phi site-to-site model for th Central & Eastern US
PHI_S2SS_MODEL = {"cena": PHI_S2SS_CENA}


def get_phi_s2ss_at_quantile(phi_model, quantile):
    """
    Returns the phi_s2ss value for all periods at the specific quantile as
    an instance of `class`::openquake.hazardlib.gsim.base.CoeffsTable
    """
    from copy import deepcopy

    # Setup SA coeffs - the backward compatible Python 2.7 way
    coeffs = deepcopy(phi_model.sa_coeffs)
    coeffs.update(phi_model.non_sa_coeffs)
    for imt in coeffs:
        if quantile is None:
            coeffs[imt] = {"phi_s2ss": phi_model[imt]["mean"]}
        else:
            coeffs[imt] = {"phi_s2ss": _at_percentile(phi_model[imt]["mean"],
                                                      phi_model[imt]["var"],
                                                      quantile)}
    return CoeffsTable.fromdict(coeffs)


def get_hard_rock_mean(self, mag, ctx, imt):
    """
    Returns the mean and standard deviations for the reference very hard
    rock condition (Vs30 = 3000 m/s)
    """
    # return Distance Tables
    imls = self.mean_table['%.2f' % mag, imt.string]
    # Get distance vector for the given magnitude
    idx = np.searchsorted(self.m_w, mag)
    dists = self.distances[:, 0, idx - 1]
    dst = getattr(ctx, self.distance_type)
    # get log(mean)
    return np.log(_get_mean(self.kind, imls, dst, dists))


def _get_f760(C_F760, vs30, CONSTANTS, is_stddev=False):
    """
    Returns very hard rock to hard rock (Vs30 760 m/s) adjustment factor
    taken as the Vs30-dependent weighted mean of two reference condition
    factors: for impedence and for gradient conditions. The weighting
    model is described by equations 5 - 7 of Stewart et al. (2019)
    """
    wimp = (CONSTANTS["wt1"] - CONSTANTS["wt2"]) *\
        (np.log(vs30 / CONSTANTS["vw2"])
         / np.log(CONSTANTS["vw1"] / CONSTANTS["vw2"])) + CONSTANTS["wt2"]
    wimp[vs30 >= CONSTANTS["vw1"]] = CONSTANTS["wt1"]
    wimp[vs30 < CONSTANTS["vw2"]] = CONSTANTS["wt2"]
    wgr = 1.0 - wimp
    if is_stddev:
        return wimp * C_F760["f760is"] + wgr * C_F760["f760gs"]
    else:
        return wimp * C_F760["f760i"] + wgr * C_F760["f760g"]


def _get_fv(C_LIN, sites, f760, CONSTANTS):
    """
    Returns the Vs30-dependent component of the mean linear amplification
    model, as defined in equation 3 of Stewart et al. (2019)
    """
    const1 = C_LIN["c"] * np.log(C_LIN["v1"] / CONSTANTS["vref"])
    const2 = C_LIN["c"] * np.log(C_LIN["v2"] / CONSTANTS["vref"])
    f_v = C_LIN["c"] * np.log(sites.vs30 / CONSTANTS["vref"])
    f_v[sites.vs30 <= C_LIN["v1"]] = const1
    f_v[sites.vs30 > C_LIN["v2"]] = const2
    idx = sites.vs30 > CONSTANTS["vU"]
    if np.any(idx):
        const3 = np.log(3000. / CONSTANTS["vU"])
        f_v[idx] = const2 - (const2 + f760[idx]) *\
            (np.log(sites.vs30[idx] / CONSTANTS["vU"]) / const3)
    idx = sites.vs30 >= 3000.
    if np.any(idx):
        f_v[idx] = -f760[idx]
    return f_v + f760


def get_fnl(C_NL, pga_rock, vs30, period):
    """
    Returns the nonlinear mean amplification according to equation 2
    of Hashash et al. (2019)
    """
    if period <= 0.4:
        vref = 760.
    else:
        vref = 3000.
    f_nl = np.zeros(vs30.shape)
    f_rk = np.log((pga_rock + C_NL["f3"]) / C_NL["f3"])
    idx = vs30 < C_NL["Vc"]
    if np.any(idx):
        # f2 term of the mean nonlinear amplification model
        # according to equation 3 of Hashash et al., (2019)
        c_vs = np.copy(vs30[idx])
        c_vs[c_vs > vref] = vref
        f_2 = C_NL["f4"] * (np.exp(C_NL["f5"] * (c_vs - 360.))
                            - np.exp(C_NL["f5"] * (vref - 360.)))
        f_nl[idx] = f_2 * f_rk[idx]
    return f_nl, f_rk


def get_linear_stddev(C_LIN, vs30, CONSTANTS):
    """
    Returns the standard deviation of the linear amplification function,
    as defined in equation 4 of Stewart et al., (2019)
    """
    sigma_v = C_LIN["sigma_vc"] + np.zeros(vs30.shape)
    idx = vs30 < C_LIN["vf"]
    if np.any(idx):
        dsig = C_LIN["sigma_L"] - C_LIN["sigma_vc"]
        d_v = (vs30[idx] - CONSTANTS["vL"]) /\
            (C_LIN["vf"] - CONSTANTS["vL"])
        sigma_v[idx] = C_LIN["sigma_L"] - (2. * dsig * d_v) +\
            dsig * (d_v ** 2.)
    idx = np.logical_and(vs30 > C_LIN["v2"], vs30 <= CONSTANTS["vU"])
    if np.any(idx):
        d_v = (vs30[idx] - C_LIN["v2"]) / (CONSTANTS["vU"] - C_LIN["v2"])
        sigma_v[idx] = C_LIN["sigma_vc"] + \
            (C_LIN["sigma_U"] - C_LIN["sigma_vc"]) * (d_v ** 2.)
    idx = vs30 >= CONSTANTS["vU"]
    if np.any(idx):
        sigma_v[idx] = C_LIN["sigma_U"] *\
            (1. - (np.log(vs30[idx] / CONSTANTS["vU"])
                   / np.log(3000. / CONSTANTS["vU"])))
    sigma_v[vs30 > 3000.] = 0.0
    return sigma_v


def get_nonlinear_stddev(C_NL, vs30):
    """
    Returns the standard deviation of the nonlinear amplification function,
    as defined in equation 2.5 of Hashash et al. (2017)
    """
    sigma_f2 = np.zeros(vs30.shape)
    sigma_f2[vs30 < 300.] = C_NL["sigma_c"]
    idx = np.logical_and(vs30 >= 300, vs30 < 1000)
    if np.any(idx):
        sigma_f2[idx] = (-C_NL["sigma_c"] / np.log(1000. / 300.)) *\
            np.log(vs30[idx] / 300.) + C_NL["sigma_c"]
    return sigma_f2


def get_site_amplification_sigma(self, sites, f_rk, C_LIN, C_F760, C_NL):
    """
    Returns the epistemic uncertainty on the site amplification factor
    """
    # In the case of the linear model sigma_f760 and sigma_fv are
    # assumed independent and the resulting sigma_flin is the root
    # sum of squares (SRSS)
    f760_stddev = _get_f760(C_F760, sites.vs30,
                            self.CONSTANTS, is_stddev=True)
    f_lin_stddev = np.sqrt(
        f760_stddev ** 2.
        + get_linear_stddev(C_LIN, sites.vs30, self.CONSTANTS) ** 2)
    # Likewise, the epistemic uncertainty on the linear and nonlinear
    # model are assumed independent and the SRSS is taken
    f_nl_stddev = get_nonlinear_stddev(C_NL, sites.vs30) * f_rk
    return np.sqrt(f_lin_stddev ** 2. + f_nl_stddev ** 2.)


def get_site_amplification(self, imt, pga_r, sites):
    """
    Returns the sum of the linear (Stewart et al., 2019) and non-linear
    (Hashash et al., 2019) amplification terms
    """
    # Get the coefficients for the IMT
    C_LIN = self.COEFFS_LINEAR[imt]
    C_F760 = self.COEFFS_F760[imt]
    C_NL = self.COEFFS_NONLINEAR[imt]
    if str(imt).startswith("PGA"):
        period = 0.01
    elif str(imt).startswith("PGV"):
        period = 0.5
    else:
        period = imt.period
    # Get f760
    f760 = _get_f760(C_F760, sites.vs30, self.CONSTANTS)
    # Get the linear amplification factor
    f_lin = _get_fv(C_LIN, sites, f760, self.CONSTANTS)
    # Get the nonlinear amplification from Hashash et al., (2017)
    f_nl, f_rk = get_fnl(C_NL, pga_r, sites.vs30, period)
    # Mean amplification
    ampl = f_lin + f_nl

    # If an epistemic uncertainty is required then retrieve the epistemic
    # sigma of both models and multiply by the input epsilon
    if self.site_epsilon:
        site_epistemic = get_site_amplification_sigma(
            self, sites, f_rk, C_LIN, C_F760, C_NL)
        ampl += self.site_epsilon * site_epistemic
    return ampl


def get_mean_amp(self, mag, ctx, imt):
    # Get the PGA on the reference rock condition
    if PGA in self.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
        rock_imt = PGA()
    else:
        rock_imt = SA(0.01)
    pga_r = get_hard_rock_mean(self, mag, ctx, rock_imt)

    # Get the desired spectral acceleration on rock
    if imt.string != "PGA":
        # Calculate the ground motion at required spectral period for
        # the reference rock
        mean = get_hard_rock_mean(self, mag, ctx, imt)
    else:
        # Avoid re-calculating PGA if that was already done!
        mean = np.copy(pga_r)

    amp = get_site_amplification(self, imt, np.exp(pga_r), ctx)
    mean += amp

    return mean, amp, pga_r


def _get_total_sigma(self, imt, mag):
    """
    Returns the estimated total standard deviation for a given intensity
    measure type and magnitude
    """
    [mag] = np.unique(np.round(mag, 2))  # by construction
    C = self.SIGMA[imt]
    if mag <= self.magnitude_limits[0]:
        # The CENA constant model is always returned here
        return C[self.tau_keys[0]]
    elif mag > self.magnitude_limits[-1]:
        return C[self.tau_keys[-1]]
    else:
        # Needs interpolation
        for i in range(len(self.tau_keys) - 1):
            l_m = self.magnitude_limits[i]
            u_m = self.magnitude_limits[i + 1]
            if mag > l_m and mag <= u_m:
                return ITPL(mag,
                            C[self.tau_keys[i + 1]],
                            C[self.tau_keys[i]],
                            l_m,
                            u_m - l_m)


def get_stddevs(self, mag, imt):
    """
    Returns the standard deviations for either the ergodic or
    non-ergodic models
    """
    if self.__class__.__name__.endswith('TotalSigma'):
        return [_get_total_sigma(self, imt, mag), 0., 0.]

    tau = _get_tau(self, imt, mag)
    phi = _get_phi(self, imt, mag)
    sigma = np.sqrt(tau ** 2 + phi ** 2)
    return [sigma, tau, phi]


def _get_tau(self, imt, mag):
    """
    Returns the inter-event standard deviation (tau)
    """
    return TAU_EXECUTION[self.tau_model](imt, mag, self.TAU)


def get_phi_ss(imt, mag, params):
    """
    Returns the single station phi (or it's variance) for a given magnitude
    and intensity measure type according to equation 5.14 of Al Atik (2015)
    """
    C = params[imt]
    phi = C["a"] + (mag - 5.0) * (C["b"] - C["a"]) / 1.5
    phi[mag <= 5.0] = C["a"]
    phi[mag > 6.5] = C["b"]
    return phi


def _get_phi(self, imt, mag):
    """
    Returns the within-event standard deviation (phi)
    """
    phi = get_phi_ss(imt, mag, self.PHI_SS)
    if self.ergodic:
        C = self.PHI_S2SS[imt]
        phi = np.sqrt(phi ** 2. + C["phi_s2ss"] ** 2.)
    return phi


class NGAEastGMPE(GMPETable):
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RotD50
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}
    # Requires Vs30 only - common to all models
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    kind = "nga_east"

    # Seven constants: vref, vL, vU, vw1, vw2, wt1 and wt2
    CONSTANTS = {"vref": 760., "vL": 200., "vU": 2000.0,
                 "vw1": 600.0, "vw2": 400.0, "wt1": 0.767, "wt2": 0.1}

    # Coefficients for the linear model, taken from the electronic supplement
    # to Stewart et al., (2017)
    COEFFS_LINEAR = CoeffsTable(sa_damping=5, table="""\
    imt           c      v1       v2      vf  sigma_vc  sigma_L  sigma_U
    pgv      -0.449   331.0    760.0   314.0     0.251    0.306    0.334
    pga      -0.290   319.0    760.0   345.0     0.300    0.345    0.480
    0.010    -0.290   319.0    760.0   345.0     0.300    0.345    0.480
    0.020    -0.303   319.0    760.0   343.0     0.290    0.336    0.479
    0.030    -0.315   319.0    810.0   342.0     0.282    0.327    0.478
    0.050    -0.344   319.0   1010.0   338.0     0.271    0.308    0.476
    0.075    -0.348   319.0   1380.0   334.0     0.269    0.285    0.473
    0.100    -0.372   317.0   1900.0   319.0     0.270    0.263    0.470
    0.150    -0.385   302.0   1500.0   317.0     0.261    0.284    0.402
    0.200    -0.403   279.0   1073.0   314.0     0.251    0.306    0.334
    0.250    -0.417   250.0    945.0   282.0     0.238    0.291    0.357
    0.300    -0.426   225.0    867.0   250.0     0.225    0.276    0.381
    0.400    -0.452   217.0    843.0   250.0     0.225    0.275    0.381
    0.500    -0.480   217.0    822.0   280.0     0.225    0.311    0.323
    0.750    -0.510   227.0    814.0   280.0     0.225    0.330    0.310
    1.000    -0.557   255.0    790.0   300.0     0.225    0.377    0.361
    1.500    -0.574   276.0    805.0   300.0     0.242    0.405    0.375
    2.000    -0.584   296.0    810.0   300.0     0.259    0.413    0.388
    3.000    -0.588   312.0    820.0   313.0     0.306    0.410    0.551
    4.000    -0.579   321.0    821.0   322.0     0.340    0.405    0.585
    5.000    -0.558   324.0    825.0   325.0     0.340    0.409    0.587
    7.500    -0.544   325.0    820.0   328.0     0.345    0.420    0.594
    10.00    -0.507   325.0    820.0   330.0     0.350    0.440    0.600
    """)

    # Coefficients for the nonlinear model, taken from Table 2.1 of
    # Hashash et al., (2017)
    COEFFS_NONLINEAR = CoeffsTable(sa_damping=5, table="""\
    imt          f3         f4         f5     Vc   sigma_c
    pgv     0.06089   -0.08344   -0.00667   2257.0   0.120
    pga     0.07520   -0.43755   -0.00131   2990.0   0.120
    0.010   0.07520   -0.43755   -0.00131   2990.0   0.120
    0.020   0.05660   -0.41511   -0.00098   2990.0   0.120
    0.030   0.10360   -0.49871   -0.00127   2990.0   0.120
    0.050   0.16781   -0.58073   -0.00187   2990.0   0.120
    0.075   0.17386   -0.53646   -0.00259   2990.0   0.120
    0.100   0.15083   -0.44661   -0.00335   2990.0   0.120
    0.150   0.14272   -0.38264   -0.00410   2335.0   0.120
    0.200   0.12815   -0.30481   -0.00488   1533.0   0.120
    0.250   0.13286   -0.27506   -0.00564   1318.0   0.135
    0.300   0.13070   -0.22825   -0.00655   1152.0   0.150
    0.400   0.09414   -0.11591   -0.00872   1018.0   0.150
    0.500   0.09888   -0.07793   -0.01028    939.0   0.150
    0.750   0.06101   -0.01780   -0.01456    835.0   0.125
    1.000   0.04367   -0.00478   -0.01823    951.0   0.060
    1.500   0.00480   -0.00086   -0.02000    882.0   0.050
    2.000   0.00164   -0.00236   -0.01296    879.0   0.040
    3.000   0.00746   -0.00626   -0.01043    894.0   0.040
    4.000   0.00269   -0.00331   -0.01215    875.0   0.030
    5.000   0.00242   -0.00256   -0.01325    856.0   0.020
    7.500   0.04219   -0.00536   -0.01418    832.0   0.020
    10.00   0.05329   -0.00631   -0.01403    837.0   0.020
    """)

    # Note that the coefficient values at 0.1 s have been smoothed with respect
    # to those needed in order to reproduce Figure 5 of Petersen et al. (2019)
    # The original f760i was 0.674 +/- 0.366, and the values below are taken
    # from the US NSHMP software
    COEFFS_F760 = CoeffsTable(sa_damping=5, table="""\
    imt       f760i     f760g   f760is   f760gs
    pgv      0.3753     0.297    0.313    0.117
    pga      0.1850     0.121    0.434    0.248
    0.010    0.1850     0.121    0.434    0.248
    0.020    0.1850     0.031    0.434    0.270
    0.030    0.2240     0.000    0.404    0.229
    0.050    0.3370     0.062    0.363    0.093
    0.075    0.4750     0.211    0.322    0.102
    0.100    0.5210     0.338    0.293    0.088
    0.150    0.5860     0.470    0.253    0.066
    0.200    0.4190     0.509    0.214    0.053
    0.250    0.3320     0.509    0.177    0.052
    0.300    0.2700     0.498    0.131    0.055
    0.400    0.2090     0.473    0.112    0.060
    0.500    0.1750     0.447    0.105    0.067
    0.750    0.1270     0.386    0.138    0.077
    1.000    0.0950     0.344    0.124    0.078
    1.500    0.0830     0.289    0.112    0.081
    2.000    0.0790     0.258    0.118    0.088
    3.000    0.0730     0.233    0.111    0.100
    4.000    0.0660     0.224    0.120    0.109
    5.000    0.0640     0.220    0.108    0.115
    7.500    0.0560     0.216    0.082    0.130
    10.00    0.0530     0.218    0.069    0.137
    """)
