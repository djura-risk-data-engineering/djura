# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from pathlib import Path
import math
import numpy as np
from scipy.interpolate import interp1d

from .utilities import find_right_index, interpolate_2d, read_json
from .constants import ESHM20_COEFFICIENTS
from . import _activation_functions as activation_functions


asset_dir = Path(__file__).resolve().parent / "assets"

AKKAR_CORRELATION_TABLE = "akkar_correlation_table.txt"

ACTIVATION_FUNCTIONS = {
    "linear": activation_functions.linear,
    "softmax": activation_functions.softmax,
    "tanh": activation_functions.tanh,
    "sigmoid": activation_functions.sigmoid,
}


def baker_jayaram(
    period1: float, period2: float,
    d1=0.366, d2=0.105, d3=0.0099, d4=0.109, d5=0.2
):
    """SA vs SA
    Valid for T = 0.01-10sec

    References
    ----------
    Baker JW, Jayaram N. Correlation of Spectral Acceleration Values
    from NGA Ground Motion Models.
    Earthquake Spectra 2008; 24(1): 299-317. DOI: 10.1193/1.2857544.

    Parameters
    ----------
    period1 : float
        First period
    period2 : float
        Second period

    Returns
    -------
    float
        Predicted correlation coefficient
    """

    period_min = min(period1, period2)
    period_max = max(period1, period2)

    c1 = 1.0 - np.cos(np.pi / 2.0 - np.log(period_max
                      / max(period_min, 0.109)) * d1)

    if period_max < d5:
        c2 = 1.0 - d2 \
            * (1.0 - 1.0 / (1.0 + np.exp(100.0 * period_max - 5.0))) \
            * (period_max - period_min) / (period_max - d3)
    else:
        c2 = 0

    if period_max < d4:
        c3 = c2
    else:
        c3 = c1

    c4 = c1 + 0.5 * (np.sqrt(c3) - c3) * \
        (1.0 + np.cos(np.pi * period_min / d4))

    if period_max <= d4:
        rho = c2
    elif period_min > d4:
        rho = c1
    elif period_max < d5:
        rho = min(c2, c4)
    else:
        rho = c4

    return min(rho, 1.0)


def akkar(period1: float, period2: float):
    """SA vs SA
    Valid for T = 0.01-4sec

    References
    ----------
    Akkar S., Sandikkaya MA., Ay BO., 2014, Compatible ground-motion
    prediction equations for damping scaling factors and vertical to
    horizontal spectral amplitude ratios for the broader Europe region,
    Bull Earthquake Eng, 12, pp. 517-547.

    Parameters
    ----------
    period1 : float
        First period
    period2 : float
        Second period

    Returns
    -------
    float
        Predicted correlation coefficient
    """
    periods = np.array(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14,
         0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3,
         0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6,
         0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.4, 1.5,
         1.6, 1.7, 1.8, 1.9, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4])

    with open(asset_dir / AKKAR_CORRELATION_TABLE, 'r') as file:
        content = file.read()

    coeff_table = np.fromstring(
        content, dtype=float, sep=" ").reshape(-1, len(periods))

    if np.any([period1, period2] < periods[0]) or \
            np.any([period1, period2] > periods[-1]):
        raise ValueError("Period array contains values outside of the "
                         "range supported by the Akkar et al. (2014) "
                         "correlation model")

    if period1 == period2:
        rho = 1.0
    else:
        rho = interpolate_2d(periods, periods, coeff_table, period1, period2)
        # rho = interpolate.interp2d(
        #     periods, periods, coeff_table, kind='linear')(
        #         period1, period2)[0]

    return rho


def bradley2011_ds() -> float:
    """Duration 575 vs Duration 595

    References
    ----------
    Bradley B.A. (2011). Correlation of significant duration with amplitude
    and cumulative intensity measures and its use in ground motion selection,
    Journal of Earthquake Engineering, 15(6): 809-832.
    DOI: 10.1080/13632469.2011.557140Correlation

    Returns
    -------
    float
        Correlation value
    """
    return 0.843


def bradley2011_ds595_sa(period: float = None) -> float:
    """Duration 595 vs SA correlation

    Lowest period is 0.01!
    Highest period is 10!

    References
    ----------
    Bradley B.A. (2011). Correlation of significant duration with amplitude
    and cumulative intensity measures and its use in ground motion selection,
    Journal of Earthquake Engineering, 15(6): 809-832.
    DOI: 10.1080/13632469.2011.557140Correlation

    Parameters
    ----------
    period : float, optional
        Period of interest, by default None (for PGA)

    Returns
    -------
    float
        Correlation value
    """

    if period is None or period < 0.01:
        return -0.405

    if not 0.01 <= period < 10:
        raise ValueError(f"Period ({period}) must be 0.01 <= x < 10")

    a0 = -0.41
    b0 = 0.01
    a1 = -0.41
    b1 = 0.04
    a2 = -0.38
    b2 = 0.08
    a3 = -0.35
    b3 = 0.26
    a4 = -0.02
    b4 = 1.40
    a5 = 0.23
    b5 = 6.00
    a6 = 0.02
    b6 = 10.0

    if period >= b0 and period < b1:
        rho_ds595_sa = a0 + (np.log(period / b0)
                             / np.log(b1 / b0)) * (a1 - a0)
    elif period >= b1 and period < b2:
        rho_ds595_sa = a1 + (np.log(period / b1)
                             / np.log(b2 / b1)) * (a2 - a1)
    elif period >= b2 and period < b3:
        rho_ds595_sa = a2 + (np.log(period / b2)
                             / np.log(b3 / b2)) * (a3 - a2)
    elif period >= b3 and period < b4:
        rho_ds595_sa = a3 + (np.log(period / b3)
                             / np.log(b4 / b3)) * (a4 - a3)
    elif period >= b4 and period < b5:
        rho_ds595_sa = a4 + (np.log(period / b4)
                             / np.log(b5 / b4)) * (a5 - a4)
    else:
        rho_ds595_sa = a5 + (np.log(period / b5)
                             / np.log(b6 / b5)) * (a6 - a5)

    return rho_ds595_sa


def bradley2011_ds575_sa(period: float = None) -> float:
    """Duration 575 vs SA correlation

    Lowest period is 0.01!
    Highest period is 10!

    References
    ----------
    Bradley B.A. (2011). Correlation of significant duration with amplitude
    and cumulative intensity measures and its use in ground motion selection,
    Journal of Earthquake Engineering, 15(6): 809-832.
    DOI: 10.1080/13632469.2011.557140Correlation

    Parameters
    ----------
    period : float, optional
        Period of interest, by default None (for PGA)

    Returns
    -------
    float
        Correlation value
    """

    if period is None or period < 0.01:
        return -0.442

    if not 0.01 <= period < 10:
        raise ValueError(f"Period ({period}) must be 0.01 <= x < 10")

    a0 = -0.45
    b0 = 0.01
    a1 = -0.39
    b1 = 0.09
    a2 = -0.39
    b2 = 0.30
    a3 = -0.06
    b3 = 1.40
    a4 = 0.16
    b4 = 6.50
    a5 = 0.00
    b5 = 10.00

    if period >= b0 and period < b1:
        rho_ds575_sa = a0 + (np.log(period / b0)
                             / np.log(b1 / b0)) * (a1 - a0)
    elif period >= b1 and period < b2:
        rho_ds575_sa = a1 + (np.log(period / b1)
                             / np.log(b2 / b1)) * (a2 - a1)
    elif period >= b2 and period < b3:
        rho_ds575_sa = a2 + (np.log(period / b2)
                             / np.log(b3 / b2)) * (a3 - a2)
    elif period >= b3 and period < b4:
        rho_ds575_sa = a3 + (np.log(period / b3)
                             / np.log(b4 / b3)) * (a4 - a3)
    else:
        rho_ds575_sa = a4 + (np.log(period / b4)
                             / np.log(b5 / b4)) * (a5 - a4)
    return rho_ds575_sa


def bradley2011_ds595_pgv() -> float:
    """Duration 595 vs PGV

    References
    ----------
    Bradley B.A. (2011). Correlation of significant duration with amplitude
    and cumulative intensity measures and its use in ground motion selection,
    Journal of Earthquake Engineering, 15(6): 809-832.
    DOI: 10.1080/13632469.2011.557140Correlation

    Returns
    -------
    float
        Correlation value
    """
    return -0.211


def bradley2011_ds575_pgv() -> float:
    """Duration 575 vs PGV

    References
    ----------
    Bradley B.A. (2011). Correlation of significant duration with amplitude
    and cumulative intensity measures and its use in ground motion selection,
    Journal of Earthquake Engineering, 15(6): 809-832.
    DOI: 10.1080/13632469.2011.557140Correlation

    Returns
    -------
    float
        Correlation value
    """
    return -0.259


def bradley2011_pga(period: float) -> float:
    """PGA vs SA correlation

    Lowest period is 0.01!
    Highest period is 10!

    References
    ----------
    Bradley, B.A. (2011). Empirical correlation of PGA, spectral accelerations
    and spectrum intensities from active shallow crustal earthquakes.
    Earthquake Engineering & Structural Dynamics, 40.
    DOI: 10.1002/eqe.1110

    Parameters
    ----------
    period : float
        Period of interest

    Returns
    -------
    float
        Correlation value
    """
    if period is None or period < 0.01:
        return 1.0

    if not 0.01 <= period < 10:
        raise ValueError(f"Period ({period}) must be 0.01 <= x < 10")

    e = [0.01, 0.20, 10]
    a = [1.00, 0.97]
    b = [0.895, 0.25]
    c = [0.06, 0.80]
    d = [1.60, 0.80]

    idx = find_right_index(e, period) - 1

    rho = (a[idx] + b[idx]) / 2 - \
        (a[idx] - b[idx]) / 2 * math.tanh(d[idx] * np.log(period / c[idx]))

    return rho


def bradley2012_pgv(period: float = None) -> float:
    """PGV vs SA correlation and PGV vs PGA correlation

    For vs SA correlation:
    Lowest period is 0.01!
    Highest period is 10!

    References
    ----------
    Bradley, B.A. (2012). Empirical Correlations between Peak Ground Velocity
    and Spectrum-Based Intensity Measures. Earthquake Spectra, 28, 17 - 35.
    DOI:10.1193/1.3675582

    Parameters
    ----------
    period : float, optional
        Period of interest, by default None (for PGA)

    Returns
    -------
    float
        Correlation value
    """

    if period is None or period < 0.01:
        # Median value of PGV vs PGA correlation
        return 0.733

    if not 0.01 <= period < 10:
        raise ValueError(f"Period ({period}) must be 0.01 <= x < 10")

    e = [0.01, 0.1, 0.75, 2.5, 10.]
    a = [0.73, 0.54, 0.80, 0.76]
    b = [0.54, 0.81, 0.76, 0.70]
    c = [0.045, 0.28, 1.10, 5.00]
    d = [1.80, 1.50, 3.00, 3.20]

    idx = find_right_index(e, period) - 1

    rho = (a[idx] + b[idx]) / 2 - \
        (a[idx] - b[idx]) / 2 * math.tanh(d[idx] * np.log(period / c[idx]))

    return rho


def dm18(period1: float, period2: float) -> float:
    """Sa_avg3 vs Sa_avg3 correlation

    For periods between 0.1 and 3 seconds

    Parameters
    ----------
    period1 : float
        First period
    period2 : float
        Second period

    References
    ----------
    Héctor Dávalos & Eduardo Miranda (2018): A Ground Motion Prediction Model
    for Average Spectral Acceleration, Journal of Earthquake Engineering,
    DOI: 10.1080/13632469.2018.1518278

    Returns
    -------
    float
        Correlation value
    """

    if not 0.1 <= period1 <= 3:
        raise ValueError(f"Period ({period1}) must be 0.1 <= x <= 3")

    if not 0.1 <= period2 <= 3:
        raise ValueError(f"Period ({period2}) must be 0.1 <= x <= 3")

    a = -0.0047
    b = 0.1608
    c = 0.4763
    d = 0.4853
    e = -0.1731

    x = np.log(period1) + np.log(period2)
    y = np.log(period1)**2 + np.log(period2)**2
    z = np.log(period1) * np.log(period2)

    rho = (1 + a * x + b * y + c * z) / \
        (1 + a * x + d * y + e * z)

    return rho


def ann_corr(im_pair: str, period1: float = None,
             period2: float = None) -> float:
    """Correlation matrices predicted through an ANN model

    Parameters
    ----------
    im_pair : str
        IMi-IMj pair
    period1 : float, optional
        Period associated with IMi, by default None
    period2 : float, optional
        Period associated with IMj, by default None

    Returns
    -------
    float
        Correlation value
    """
    CORRELATIONS_ANN = read_json(asset_dir / "correlation_models.json")

    imi, imj = im_pair.split("-")

    try:
        im_pair = f"{imi}-{imj}"
        corr = CORRELATIONS_ANN[f"corr_{im_pair}"]
    except KeyError:
        im_pair = f"{imj}-{imi}"
        corr = CORRELATIONS_ANN[f"corr_{im_pair}"]
        # Switch positions too
        period2, period1 = period1, period2
        imj, imi = imi, imj

    corr = np.asarray(corr)

    periods_i = CORRELATIONS_ANN.get(f"T_{imi}")
    periods_j = CORRELATIONS_ANN.get(f"T_{imj}")

    if periods_i is None and periods_j is None:
        # Both IMs are period-independent
        return corr

    if periods_i is None or periods_j is None:
        # Only one IM is period-independent
        periods = periods_i or periods_j
        period = period1 or period2

        corr = corr.T

        if period < periods[0]:
            return corr[0]
        if period > periods[-1]:
            return corr[-1]

        interp = interp1d(periods, corr)
        return interp(period)

    if imi == imj and period1 == period2:
        return 1.0

    # Both IMs are period-dependent
    _val = interpolate_2d(periods_i, periods_j, corr, period1, period2)
    return np.atleast_1d(_val)


def aso2024(im_pair: str, period1: float = None,
            period2: float = None) -> float:
    """Correlation matrices predicted through an ANN model

    Parameters
    ----------
    im_pair : str
        IMi-IMj pair
    period1 : float, optional
        Period associated with IMi, by default None
    period2 : float, optional
        Period associated with IMj, by default None

    Returns
    -------
    float
        Correlation value
    """
    MODELS_ANN = read_json(asset_dir / "corr_ann.json")

    def _generate_function(x, biases, weights):
        biases = np.asarray(biases)
        weights = np.asarray(weights).T

        return biases.reshape(1, -1) + np.dot(weights, x.T).T

    TRANSFORMATIONS = frozenset({
        "SA-Ds595", "SA-Ds575",
        "Sa_avg2-Ds595", "Sa_avg2-Ds575", "Sa_avg2-PGA", "Sa_avg2-PGV",
        "Sa_avg3-Ds595", "Sa_avg3-Ds575", "Sa_avg3-PGA", "Sa_avg3-PGV",
    })

    imi, imj = im_pair.split("-")

    try:
        im_pair = f"{imi}-{imj}"
        model = MODELS_ANN[im_pair]
    except KeyError:
        im_pair = f"{imj}-{imi}"
        model = MODELS_ANN[im_pair]
        # Switch positions too
        period2, period1 = period1, period2
        imj, imi = imi, imj

    if period1 is None or period2 is None:
        # Only one IM is period-independent
        period = period1 or period2

        x = np.array([period])

    elif imi == imj:
        period_min = min(period1, period2)
        period_max = max(period1, period2)
        x = np.array([period_max, period_min])
    else:
        x = np.array([period1, period2])

    if imi == imj and period1 == period2:
        return 1.0

    biases = model["biases"]
    weights = model["weights"]
    act_funcs = model["activation-functions"]

    for i, act in enumerate(act_funcs):
        activation = ACTIVATION_FUNCTIONS[act]

        if im_pair in TRANSFORMATIONS and i == 0:
            x = np.log(x)

        _data = _generate_function(x, biases[i], weights[i])
        x = activation(_data)

    if isinstance(x[0], float):
        return x[0]
    else:
        return x[0][0]


def eshm20(period1: float, period2: float):
    d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["total"]

    return baker_jayaram(period1, period2, d1, d2, d3, d4, d5)


def eshm20_between_event(period1: float, period2: float):
    d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["between-event"]

    return baker_jayaram(period1, period2, d1, d2, d3, d4, d5)


def eshm20_between_site(period1: float, period2: float):
    d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["between-site"]

    return baker_jayaram(period1, period2, d1, d2, d3, d4, d5)


def eshm20_within_event(period1: float, period2: float):
    d1, d2, d3, d4, d5 = ESHM20_COEFFICIENTS["within-event"]

    return baker_jayaram(period1, period2, d1, d2, d3, d4, d5)


def baker2007_ia_sa(period: float):
    """
    Arias Intensity (IA) vs Spectral acceleration correlation

    References
    ----------
    Baker, J.W. (2007). Correlation of ground motion intensity parameters
    used for predicting structural and geaotehnical response.
    Applications of Statistics and Probability in Civil Engineering.
    DOI:10.1017/CBO9780511509759.001

    Parameters
    ----------
    period : float, optional
        Period of interest, by default None (for PGA)

    Returns
    -------
    float
        Correlation value
    """

    if period < 0.11:   # Min 0.05
        return 0.344 - 0.152 * np.log(period)
    elif 0.11 <= period < 0.4:
        return 0.971 + 0.131 * np.log(period)
    elif 0.4 <= period:    # max 5
        return 0.697 - 0.166 * np.log(period)


def bradley2015_ia_sa(period: float):
    """
    Piecewise median correlation between Arias Intensity (IA) vs
    Spectral acceleration correlation

    References
    ----------
    Bradley, B. A. (2015). Correlation of Arias intensity with amplitude,
    duration and cumulative intensity measures. Soil Dynamics and Earthquake
    Engineering, 78, 89-98. https://doi.org/10.1016/j.soildyn.2015.07.009

    Parameters
    ----------
    period : float, optional
        Period of interest, by default None (for PGA)

    Returns
    -------
    float
        Correlation value
    """

    if 0.01 <= period < 0.20:
        a, b, c, d = 0.83, 0.74, 0.05, 2.5
    elif 0.20 <= period < 4.0:
        a, b, c, d = 0.74, 0.46, 1.0, 1.5
    elif 4.0 <= period:     # max 10s
        a, b, c, d = 0.46, 0.35, 5.5, 5.6
    else:
        # Anything below 0.01, assume PGA
        return bradley2015_ia_pga()

    return (a + b) / 2 - (a - b) / 2 * np.tanh(d * np.log(period / c))


def bradley2015_ia_pga():
    """
    Aris Intensity (IA) vs PGA

    References
    ----------
    Bradley, B. A. (2015). Correlation of Arias intensity with amplitude,
    duration and cumulative intensity measures. Soil Dynamics and Earthquake
    Engineering, 78, 89-98. https://doi.org/10.1016/j.soildyn.2015.07.009

    Returns
    -------
    float
        Correlation value
    """
    return 0.83


def bradley2015_ia_pgv():
    """
    Aris Intensity (IA) vs PGV

    References
    ----------
    Bradley, B. A. (2015). Correlation of Arias intensity with amplitude,
    duration and cumulative intensity measures. Soil Dynamics and Earthquake
    Engineering, 78, 89-98. https://doi.org/10.1016/j.soildyn.2015.07.009

    Returns
    -------
    float
        Correlation value
    """
    return 0.73


def bradley2015_ia_ds575():
    """
    Aris Intensity (IA) vs Ds575

    References
    ----------
    Bradley, B. A. (2015). Correlation of Arias intensity with amplitude,
    duration and cumulative intensity measures. Soil Dynamics and Earthquake
    Engineering, 78, 89-98. https://doi.org/10.1016/j.soildyn.2015.07.009

    Returns
    -------
    float
        Correlation value
    """
    return -0.19


def bradley2015_ia_ds595():
    """
    Aris Intensity (IA) vs Ds595

    References
    ----------
    Bradley, B. A. (2015). Correlation of Arias intensity with amplitude,
    duration and cumulative intensity measures. Soil Dynamics and Earthquake
    Engineering, 78, 89-98. https://doi.org/10.1016/j.soildyn.2015.07.009

    Returns
    -------
    float
        Correlation value
    """
    return -0.20


def _kohrangi2020_sa_im(T, a, b):
    """Piecewise log-linear SA-vs-IM correlation — Eq. (12), Table 1.

    Interpolates the correlation linearly in ln(T) between the tabulated
    anchor points ``(b_i, a_i)`` of Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s); must lie within ``[b[0], b[-1]]``.
    a : float or array_like
        Correlation values at the anchor periods.
    b : float or array_like
        Anchor periods (s), strictly increasing.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SA(T), IM].

    Reference
    ---------
    Kohrangi, M., Papadopoulos, A. N., Bazzurro, P., and Vamvatsikos, D.
    (2020). "Correlation of spectral acceleration values of vertical and
    horizontal ground motion pairs." Earthquake Spectra, 36(4).
    DOI: 10.1177/8755293020919416
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    T = np.asarray(T, dtype=float)
    if np.any((T < b[0]) | (T > b[-1])):
        raise ValueError(
            f"Period(s) outside the model range [{b[0]}, {b[-1]}] s."
        )
    # Piecewise linear interpolation in ln(T)
    rho = np.interp(np.log(T), np.log(b), a)
    return rho if rho.ndim else float(rho)


def kohrangi2020_sav_ds575(T):
    """
    Vertical spectral acceleration SAV(T) vs significant duration Ds5-75
    correlation — Eq. (12) and Table 1 ('SAV-Ds5-75') of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(T), Ds5-75].
    """
    a = [-0.299, -0.306, -0.216, -0.362, 0.154, -0.016]
    b = [0.01, 0.02, 0.06, 0.33, 4.46, 10.00]
    return _kohrangi2020_sa_im(T, a, b)


def kohrangi2020_sav_ds595(T):
    """
    Vertical spectral acceleration SAV(T) vs significant duration Ds5-95
    correlation — Eq. (12) and Table 1 ('SAV-Ds5-95') of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(T), Ds5-95].
    """
    a = [-0.269, -0.278, -0.201, -0.311, 0.186, 0.014]
    b = [0.01, 0.02, 0.05, 0.34, 4.08, 10.00]
    return _kohrangi2020_sa_im(T, a, b)


def kohrangi2020_sav_pgav(T):
    """
    Vertical spectral acceleration SAV(T) vs horizontal peak ground
    velocity PGVH correlation — Eq. (12) and Table 1 ('SAV-PGVH') of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(T), PGVH].
    """
    a = [0.447, 0.432, 0.263, 0.587, 0.602, 0.371]
    b = [0.01, 0.02, 0.06, 0.42, 1.45, 10.00]
    return _kohrangi2020_sa_im(T, a, b)


def kohrangi2020_sav_pgvv(T):
    """
    Vertical spectral acceleration SAV(T) vs vertical peak ground velocity
    PGVV correlation — Eq. (12) and Table 1 ('SAV-PGVV') of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(T), PGVV].
    """
    a = [0.553, 0.534, 0.333, 0.623, 0.732, 0.601]
    b = [0.01, 0.02, 0.06, 0.40, 1.62, 10.00]
    return _kohrangi2020_sa_im(T, a, b)


def kohrangi2020_sah_pgvv(T):
    """
    Horizontal spectral acceleration SAH(T) vs vertical peak ground
    velocity PGVV correlation — Eq. (12) and Table 1 ('SAH-PGVV') of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        SA vibration period (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAH(T), PGVV].
    """
    a = [0.473, 0.492, 0.276, 0.551, 0.623, 0.550]
    b = [0.01, 0.02, 0.11, 0.55, 1.77, 10.00]
    return _kohrangi2020_sa_im(T, a, b)


def kohrangi2020_sav_sav(T1, T2):
    """
    Cross-correlation between vertical spectral accelerations at periods
    T1 and T2 — Eqs. (8)-(11) of Kohrangi et al. (2020). The functional
    forms follow Baker and Jayaram (2008), refitted to vertical-component
    residuals from NGA-West2.

    Parameters
    ----------
    T1, T2 : float or array_like
        Periods of the two vertical-component SAs (s), 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(T1), SAV(T2)] (symmetric).
    """
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)
    Tmin = np.minimum(T1, T2)
    Tmax = np.maximum(T1, T2)

    # Eq. (9)
    C1 = 1.0 - np.cos(
        np.pi / 2.0
        - 0.384 * np.log(Tmax / np.maximum(Tmin, 0.072))
    )

    # Eq. (10)
    with np.errstate(over="ignore"):
        C2 = np.where(
            Tmax < 0.2,
            1.0
            - 0.166
            * (1.0 - 1.0 / (1.0 + np.exp(100.0 * Tmax - 3.745)))
            * ((Tmax - Tmin) / (Tmax - 0.0067)),
            0.0,
        )

    # Eq. (11)
    C3 = C1 + 0.5 * (np.sqrt(C1) - C1) * (
        1.0 + np.cos(np.pi * Tmin / 0.072)
    )

    # Eq. (8) — case selection
    rho = np.where(
        Tmax < 0.072,
        C2,
        np.where(
            Tmin > 0.072,  # (and Tmax >= 0.072)
            C1,
            np.where(
                (Tmax >= 0.072) & (Tmax < 0.2) & (Tmin <= 0.072),
                np.minimum(C2, C3),
                C3,
            ),
        ),
    )
    return rho if rho.ndim else float(rho)


def _kohrangi2020_rho0(T):
    """
    Same-period correlation between vertical and horizontal spectral
    accelerations, rho0(T) = rho[SAV(T), SAH(T)] — Eq. (5) of
    Kohrangi et al. (2020).

    Parameters
    ----------
    T : float or array_like
        Vibration period (s), valid for 0.01 <= T <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho0(T).
    """
    T = np.asarray(T, dtype=float)
    rho = np.where(
        T <= 2.23,
        0.70 + 0.028 * np.cos(
            np.pi / 2.0
            + np.pi * np.log(np.maximum(T, 0.036) / 0.14) / 2.76
        ),
        0.60 + 0.10 * np.cos(0.62 * np.pi * np.log(T / 2.23)),
    )
    return rho if rho.ndim else float(rho)


def _kohrangi2020_c_coefficient(Tmax):
    """Fitted coefficient c — Eq. (7) of Kohrangi et al. (2020)."""
    return 1.0 - np.cos(
        (np.pi / 3.67)
        * np.minimum(2.94, np.log(np.maximum(Tmax / 0.10, 1.0)))
    )


def kohrangi2020_sav_sah(TV, TH):
    """
    Cross-correlation between vertical spectral acceleration at period TV
    and horizontal spectral acceleration at period TH — Eqs. (6)-(7) of
    Kohrangi et al. (2020).

    The parametric model is symmetric in (TV, TH) by construction (a
    modeling choice made by the authors to avoid overly complex equations,
    despite slight asymmetry in the empirical estimates).

    Parameters
    ----------
    TV : float or array_like
        Period of the vertical component SA (s), 0.01 <= TV <= 10.
    TH : float or array_like
        Period of the horizontal component SA (s), 0.01 <= TH <= 10.

    Returns
    -------
    np.ndarray or float
        Correlation coefficient rho[SAV(TV), SAH(TH)].
    """
    TV = np.asarray(TV, dtype=float)
    TH = np.asarray(TH, dtype=float)
    Tmin = np.minimum(TV, TH)
    Tmax = np.maximum(TV, TH)

    r0 = _kohrangi2020_rho0(Tmax)
    c = _kohrangi2020_c_coefficient(Tmax)

    # Branch Tmin > 0.10 s
    with np.errstate(divide="ignore", invalid="ignore"):
        branch_hi = r0 * (
            1.0
            - 0.25 * c
            + 0.25 * c * np.cos(
                np.pi * np.log(Tmin / Tmax) / np.log(Tmax / 0.10)
            )
        )

        # Branch Tmin <= 0.10 s
        branch_lo = r0 * (
            1.0
            - 0.44 * c
            - 0.06 * c * np.cos(
                np.pi
                * np.log(np.maximum(Tmin / 0.10, 0.014))
                / np.log(np.maximum(Tmax / 0.10, 7.14))
            )
        )

    # When Tmin == Tmax (same period), Eq. 6 upper branch gives
    # cos(0/0) -> handle explicitly: rho = rho0(T).
    same = np.isclose(Tmin, Tmax)
    rho = np.where(Tmin > 0.10, branch_hi, branch_lo)
    rho = np.where(same & (Tmin > 0.10), r0, rho)

    return rho if rho.ndim else float(rho)
