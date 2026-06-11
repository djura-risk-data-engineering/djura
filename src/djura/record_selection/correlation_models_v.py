# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Correlation models for the *vertical* component of ground motion.

Unlike the horizontal-component models in :mod:`correlation_models`, the
vertical-component total correlation is a magnitude- and region-dependent
quantity. It is obtained by blending the (period-only) within-event and
between-event correlations with the standard-deviation split of the
companion vertical GMPE, which is why ``mag`` and ``region`` are required.
"""
from pathlib import Path
import numpy as np

from .utilities import interpolate_2d
from .gsim.imt import SA
from .gsim.models.gulerce_2017 import (
    GulerceEtAl2017,
    _get_intra_event_std,
    _get_inter_event_std,
)


asset_dir = Path(__file__).resolve().parent / "assets"

#: Asset files holding Tables 4 and 5 of Gulerce et al. (2017): the
#: within-event (intra-event) and between-event (inter-event) correlations
#: of the vertical-component epsilons.
GKAS2017_WITHIN_TABLE = "gkas2017_within_table.txt"
GKAS2017_BETWEEN_TABLE = "gkas2017_between_table.txt"

#: Periods (sec) at which the GKAS (2017) correlation tables are tabulated;
#: valid input range for both periods is [0.01, 10] sec.
GKAS2017_PERIODS = np.array(
    [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
     0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10])

#: Regions supported by the vertical GMPE used for the sigma split; the same
#: codes as :class:`GulerceEtAl2017`. ``"CAL"`` is the default (global) model.
GKAS2017_REGIONS = frozenset({"CAL", "TWN", "ITA", "MID", "CHN", "JPN"})

GKAS2017_WITHIN = np.loadtxt(asset_dir / GKAS2017_WITHIN_TABLE)
GKAS2017_BETWEEN = np.loadtxt(asset_dir / GKAS2017_BETWEEN_TABLE)


def _vertical_sigmas(period: float, mag: float, region: str):
    """Standard-deviation split of the vertical GMPE at a single period.

    Reuses the coefficient table and standard-deviation model of
    :class:`GulerceEtAl2017` (the companion vertical GMPE) so the sigmas are,
    by construction, identical to those the GMPE would predict. In this model
    ``tau`` and ``phi`` depend only on magnitude and region (Equations 19-20),
    hence no distance/geometry parameters are needed.

    Parameters
    ----------
    period : float
        Period of interest (sec). Off-grid periods use the GMPE's own
        (log-period) coefficient interpolation.
    mag : float
        Moment magnitude.
    region : str
        One of :data:`GKAS2017_REGIONS`.

    Returns
    -------
    tuple of float
        ``(sigma, tau, phi)`` total, between-event and within-event sigmas.
    """
    C = GulerceEtAl2017.COEFFS[SA(period)]
    m = np.atleast_1d(float(mag))
    phi = float(_get_intra_event_std(region, C, m)[0])
    tau = float(_get_inter_event_std(region, C, m)[0])
    sigma = np.sqrt(phi ** 2 + tau ** 2)
    return sigma, tau, phi


def gkas2017_v(period1: float, period2: float, mag: float,
               region: str = "CAL"):
    """Vertical-component SA vs SA correlation.

    Valid for T = 0.01-10 sec. Provides the correlation between the epsilons
    of the vertical (V) component of ground motion at ``period1`` and
    ``period2``.

    The within-event and between-event correlations are interpolated from
    Tables 4 and 5 in log-period space. The total correlation blends them
    with the vertical GMPE's standard-deviation split:

        rho_total = (phi_a * phi_b) / (sig_a * sig_b) * rho_within
                  + (tau_a * tau_b) / (sig_a * sig_b) * rho_between

    Because that split depends on magnitude and region, ``mag`` and
    ``region`` are required for the total correlation (the within/between
    correlations themselves depend only on the two periods).

    References
    ----------
    Gulerce, Z., Kamai, R., Abrahamson, N. A., & Silva, W. J. (2017).
    Ground Motion Prediction Equations for the Vertical Ground Motion
    Component Based on the NGA-W2 Database. Earthquake Spectra, 33(2),
    499-528. DOI: 10.1193/121814eqs213m

    Adapted from the MATLAB implementation by N. Simon Kwong.

    Parameters
    ----------
    period1 : float
        First period (sec), 0.01 to 10.
    period2 : float
        Second period (sec), 0.01 to 10.
    mag : float
        Moment magnitude (used only for the total correlation).
    region : str, optional
        Region code for the GMPE sigma split, by default ``"CAL"`` (global).
        One of :data:`GKAS2017_REGIONS`.

    Returns
    -------
    tuple of float
        ``(rho_total, rho_between, rho_within)``.
    """
    lo, hi = GKAS2017_PERIODS[0], GKAS2017_PERIODS[-1]
    if not (lo <= period1 <= hi) or not (lo <= period2 <= hi):
        raise ValueError(
            f"Periods ({period1}, {period2}) must be within "
            f"[{lo}, {hi}] sec for the Gulerce et al. (2017) "
            "vertical correlation model")

    if region not in GKAS2017_REGIONS:
        raise ValueError(
            f"region '{region}' is not supported; choose one of "
            f"{sorted(GKAS2017_REGIONS)}")

    log_periods = np.log(GKAS2017_PERIODS)

    if period1 == period2:
        # Correlation is unity on the diagonal of both tables.
        rho_within = rho_between = 1.0
    else:
        # Interpolate on a semilog (log-period) scale, as in the original.
        rho_within = float(interpolate_2d(
            log_periods, log_periods, GKAS2017_WITHIN,
            np.log(period1), np.log(period2)))
        rho_between = float(interpolate_2d(
            log_periods, log_periods, GKAS2017_BETWEEN,
            np.log(period1), np.log(period2)))

    sig_a, tau_a, phi_a = _vertical_sigmas(period1, mag, region)
    sig_b, tau_b, phi_b = _vertical_sigmas(period2, mag, region)

    rho_total = (phi_a * phi_b) / (sig_a * sig_b) * rho_within \
        + (tau_a * tau_b) / (sig_a * sig_b) * rho_between

    return float(rho_total), rho_between, rho_within


def rho_HV_indirect(
    sigma_H_Tj: np.ndarray,
    sigma_VH_Tj: np.ndarray,
    rho_HH: np.ndarray,
    rho_H_VH_cross: np.ndarray,
    rho_HVH_Tj: np.ndarray,
) -> np.ndarray:
    """
    Compute the H-V correlation coefficient rho_{H,V}(Ti, Tj) between the
    horizontal spectral acceleration at period Ti and the vertical spectral
    acceleration at period Tj.

    This implements Equation (3) of Kwong et al. (2020), which derives
    rho_{H,V} from GMPMs for the horizontal component and the V/H ratio,
    avoiding the need for a direct H-V correlation model.

    Parameters
    ----------
    sigma_H_Tj : np.ndarray
        Logarithmic standard deviation of the horizontal component at
        period Tj, i.e. sigma_H(Tj). Obtained from a horizontal GMPM
        (e.g., Campbell & Bozorgnia 2014, NGA-West2).
    sigma_VH_Tj : np.ndarray
        Logarithmic standard deviation of the V/H ratio at period Tj,
        i.e. sigma_{V/H}(Tj). Obtained from a V/H GMPM
        (e.g., Bozorgnia & Campbell 2016, Earthquake Spectra 32(2), 951-978).
    rho_HH : np.ndarray
        Correlation of horizontal spectral accelerations between periods
        Ti and Tj, i.e. rho_{H,H}(Ti, Tj). Obtained from a horizontal
        correlation model (e.g., Baker & Jayaram 2008).
    rho_H_VH_cross : np.ndarray
        Correlation between the horizontal spectral acceleration at Ti and
        the V/H ratio at Tj, i.e. rho_{H,V/H}(Ti, Tj). Obtained from a
        H-to-V/H cross-correlation model
        (e.g., Gulerce & Abrahamson 2011, Earthquake Spectra 27(4), 1023-1047).
    rho_HVH_Tj : np.ndarray
        Cross-correlation between the horizontal spectral acceleration and
        the V/H ratio at the same period Tj, i.e. rho_{H,V/H}(Tj, Tj).
        This is the diagonal of the H-to-V/H cross-correlation matrix.

    Returns
    -------
    np.ndarray
        rho_{H,V}(Ti, Tj) — correlation coefficient between the horizontal
        spectral acceleration at Ti and the vertical spectral acceleration
        at Tj.

    References
    ----------
    Kwong, N. S., and Chopra, A. K. (2020).
        "Selecting, scaling, and orienting three components of ground
        motions for intensity-based assessments at far-field sites."
        Earthquake Spectra. DOI: 10.1177/8755293019899954.

    Notes
    -----
    - All inputs must be broadcastable to the same shape. Typically
      sigma arrays are 1-D over Tj, while rho arrays are 2-D (Ti x Tj).
    - rho_{H,V}(Ti, Tj) is NOT symmetric: swapping Ti and Tj in the
      input periods will yield a different result (see Fig. 2 of the paper).
    - When a direct H-V correlation model is unavailable (e.g., outside
      the 0.05-4 s range of Baker & Cornell 2006), this equation provides
      an alternative derivation over a wider period range.
    """
    numerator = (
        sigma_H_Tj * rho_HH
        + sigma_VH_Tj * rho_H_VH_cross
    )

    denominator_sq = (
        sigma_H_Tj ** 2
        + sigma_VH_Tj ** 2
        + 2.0 * sigma_H_Tj * sigma_VH_Tj * rho_HVH_Tj
    )

    return numerator / np.sqrt(denominator_sq)


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
