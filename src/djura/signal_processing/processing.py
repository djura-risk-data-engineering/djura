# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Signal processing tools."""
from typing import Literal, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, windows

#: Polynomial degree removed by each baseline-correction type.
BASELINE_DEGREES = {"Constant": 0, "Linear": 1, "Quadratic": 2, "Cubic": 3}


def baseline_correction(
        values: np.ndarray, dt: float,
        polynomial_type: Literal["Constant", "Linear", "Quadratic",
                                 "Cubic"]) -> np.ndarray:
    """Remove a best-fit polynomial trend from a signal.

    References
    ----------
    Kramer, S. L. (1996). Geotechnical Earthquake Engineering. Prentice Hall.

    Parameters
    ----------
    values : numpy.ndarray
        Input signal.
    dt : float
        Sampling interval [s].
    polynomial_type : {'Constant', 'Linear', 'Quadratic', 'Cubic'}
        Degree of the trend to remove.

    Returns
    -------
    numpy.ndarray
        Corrected signal.

    Raises
    ------
    ValueError
        On an unrecognised ``polynomial_type``.
    """
    if polynomial_type not in BASELINE_DEGREES:
        raise ValueError("polynomial_type must be one of %s; got %r"
                         % (sorted(BASELINE_DEGREES), polynomial_type))
    values = np.asarray(values)
    time = np.linspace(0, (values.size - 1) * dt, values.size)
    trend = np.polyval(np.polyfit(time, values, BASELINE_DEGREES[
        polynomial_type]), time)
    return values - trend


def butterworth_filter(
        values: np.ndarray, dt: float,
        cut_off: Union[float, Tuple[float, float]] = (0.1, 25),
        filter_order: int = 4,
        filter_type: Literal["lowpass", "highpass", "bandpass",
                             "bandstop"] = "bandpass",
        filtering: Literal["causal", "acausal"] = "acausal",
        alpha_window: float = 0.0) -> np.ndarray:
    """Butterworth IIR filtering, one-pass or zero-phase.

    The signal is tapered with a Tukey window, zero-padded by half its length
    at each end, filtered, and the pads removed.

    References
    ----------
    Boore, D. M., and Akkar, S. (2003). Effect of causal and acausal filters
    on elastic and inelastic response spectra. Earthquake Engineering &
    Structural Dynamics, 32, 1729-1748.
    Boore, D. M. (2005). On pads and filters: processing strong-motion data.
    Bulletin of the Seismological Society of America, 95(2), 745-750.

    Parameters
    ----------
    values : numpy.ndarray
        Input signal.
    dt : float
        Sampling interval [s].
    cut_off : float or tuple of float, optional
        Corner frequency [Hz]; scalar for lowpass and highpass, a pair for
        bandpass and bandstop. ``(0.1, 25)`` by default.
    filter_order : int, optional
        Filter order, 4 by default.
    filter_type : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        'bandpass' by default.
    filtering : {'causal', 'acausal'}, optional
        One forward pass, or forward and reverse for zero phase. 'acausal' by
        default, and preferable where usable since it adds no phase shift.
    alpha_window : float, optional
        Tukey window shape between 0 and 1, 0.0 by default.

    Returns
    -------
    numpy.ndarray
        Filtered signal.

    Raises
    ------
    ValueError
        On an unrecognised ``filtering`` or an out-of-range ``alpha_window``.
    """
    if not 0.0 <= alpha_window <= 1.0:
        raise ValueError("alpha_window must be between 0 and 1")
    if filtering not in ("causal", "acausal"):
        raise ValueError("filtering must be 'causal' or 'acausal'; got %r"
                         % filtering)

    values = np.asarray(values)
    if isinstance(cut_off, (list, tuple)):
        cut_off = np.asarray(cut_off)

    nyquist = 0.5 / dt
    numerator, denominator = butter(filter_order, cut_off / nyquist,
                                    filter_type)

    pad = round(values.size / 2)
    tapered = windows.tukey(values.size, alpha_window) * values
    padded = np.concatenate((np.zeros(pad), tapered, np.zeros(pad)))

    if filtering == "acausal":
        padded = filtfilt(numerator, denominator, padded)
    else:
        padded = lfilter(numerator, denominator, padded)
    return padded[pad:padded.size - pad]
