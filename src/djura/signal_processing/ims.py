# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Intensity measure calculators."""
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import butter, filtfilt, find_peaks, lfilter
from scipy.spatial import ConvexHull, QhullError

from .sdof import lrha, nigam_jennings
from .units import GRAVITY, G_TO_CM

#: Relative tolerance when checking that a period vector reaches the bounds of
#: a spectrum-intensity integral.
PERIOD_TOLERANCE = 1e-6


#: SDOF solver used when none is named. ``"newmark"`` is
#: :func:`djura.signal_processing.sdof.lrha`, ``"exact"`` is
#: :func:`djura.signal_processing.sdof.nigam_jennings`.
DEFAULT_SOLVER = "exact"

#: Available solvers.
SOLVERS = ("newmark", "exact")

#: Working float type by precision. float32 halves the memory; float64 is
#: better conditioned.
PRECISIONS = {32: np.float32, 64: np.float64}


#: Rotation angles for RotDxx [deg]: one degree steps over a half turn, with
#: 180 left out as the duplicate of 0. NGA-West2's convention.
DEFAULT_ANGLES = np.arange(0, 180, dtype=np.float32)

#: Sentinel distinguishing 'not given' from an explicit None, so that
#: ``min_points=None`` can mean 'do not resample'.
UNSET = object()

#: Minimum samples per oscillator period. Where the record is coarser, the
#: trace is linearly interpolated before solving, since a sampled peak
#: under-estimates the continuous one. ``None`` solves at the record's step.
MIN_POINTS_PER_PERIOD = 10

#: Bytes of working memory the chunked routines aim to stay within.
MEMORY_BUDGET = 256 * 1024 ** 2


def _integrate_range(values: np.ndarray, periods: np.ndarray, lower: float,
                     upper: float) -> float:
    """Trapezoidal integral of ``values`` over the periods in
    ``[lower, upper]``, or NaN if the vector does not reach both bounds.

    ``values`` may carry trailing axes, such as rotation angles, which are
    kept in the result.
    """
    periods = np.asarray(periods, dtype=float)
    values = np.asarray(values, dtype=float)
    tolerance = PERIOD_TOLERANCE * max(abs(upper), 1.0)
    inside = (periods >= lower - tolerance) & (periods <= upper + tolerance)
    scalar = values.ndim == 1

    def unusable():
        """NaN in the shape the caller asked for."""
        return (float("nan") if scalar
                else np.full(values.shape[1:], np.nan))

    if inside.sum() < 2:
        return unusable()
    selected = periods[inside]
    if selected[0] > lower + tolerance or selected[-1] < upper - tolerance:
        return unusable()
    out = trapezoid(values[inside], selected, axis=0)
    return float(out) if scalar else out


def peak_ground_acceleration(acc: np.ndarray) -> float:
    """Peak ground acceleration [g].

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].

    Returns
    -------
    float
        PGA [g].
    """
    return float(np.max(np.abs(np.asarray(acc))))


def velocity(acc: np.ndarray, dt: float, dtype=None) -> np.ndarray:
    """Ground velocity history [cm/s].

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].

    Returns
    -------
    numpy.ndarray
        Velocity [cm/s].
    """
    acc = np.asarray(acc)
    dtype = np.dtype(acc.dtype if dtype is None and acc.dtype.kind == "f"
                     else dtype or np.float32)
    acc = acc.astype(dtype, copy=False)
    return cumulative_trapezoid(acc * dtype.type(G_TO_CM),
                                dx=dt, initial=0.0).astype(dtype)


def displacement(vel: np.ndarray, dt: float, dtype=None) -> np.ndarray:
    """Ground displacement history [cm].

    Parameters
    ----------
    vel : numpy.ndarray
        Velocity [cm/s].
    dt : float
        Time step [s].

    Returns
    -------
    numpy.ndarray
        Displacement [cm].
    """
    vel = np.asarray(vel)
    dtype = np.dtype(vel.dtype if dtype is None and vel.dtype.kind == "f"
                     else dtype or np.float32)
    vel = vel.astype(dtype, copy=False)
    return cumulative_trapezoid(vel, dx=dt, initial=0.0).astype(dtype)


def peak_ground_velocity(vel: np.ndarray) -> float:
    """Peak ground velocity [cm/s].

    Parameters
    ----------
    vel : numpy.ndarray
        Velocity [cm/s].

    Returns
    -------
    float
        PGV [cm/s].
    """
    return float(np.max(np.abs(np.asarray(vel))))


def peak_ground_displacement(disp: np.ndarray) -> float:
    """Peak ground displacement [cm].

    Parameters
    ----------
    disp : numpy.ndarray
        Displacement [cm].

    Returns
    -------
    float
        PGD [cm].
    """
    return float(np.max(np.abs(np.asarray(disp))))


def arias_series(acc: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative Arias intensity against time [m/s].

    References
    ----------
    Arias, A. (1970). A measure of earthquake intensity. In Seismic Design for
    Nuclear Power Plants, MIT Press, 438-483.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].

    Returns
    -------
    numpy.ndarray
        Cumulative Arias intensity [m/s].
    """
    acc = np.asarray(acc, dtype=np.float64)
    return (np.pi * GRAVITY / 2.0) * np.cumsum(acc ** 2) * dt


def arias_intensity(acc: np.ndarray, dt: float) -> float:
    """Arias intensity [m/s].

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].

    Returns
    -------
    float
        Arias intensity [m/s].
    """
    return float(arias_series(acc, dt)[-1])


def cumulative_absolute_velocity(acc: np.ndarray, dt: float) -> float:
    """Cumulative absolute velocity [g.s].

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].

    Returns
    -------
    float
        CAV [g.s].
    """
    acc = np.asarray(acc, dtype=np.float64)
    return float(trapezoid(np.abs(acc), dx=dt))


def root_mean_square(series: np.ndarray, dt: float) -> float:
    """Root mean square of a series over its full length.

    Parameters
    ----------
    series : numpy.ndarray
        Acceleration [g], velocity [cm/s] or displacement [cm].
    dt : float
        Time step [s].

    Returns
    -------
    float
        RMS, in the unit of ``series``.
    """
    series = np.asarray(series, dtype=np.float64)
    total_time = dt * (series.size - 1)
    if total_time <= 0:
        return float("nan")
    return float(np.sqrt(trapezoid(series ** 2, dx=dt) / total_time))


def characteristic_intensity(arms: float, duration: float) -> float:
    """Characteristic intensity.

    Parameters
    ----------
    arms : float
        Root mean square acceleration [g].
    duration : float
        Duration [s]. The full record length is the usual choice but a
        significant duration is often the better input.

    Returns
    -------
    float
        Characteristic intensity [g**1.5 s**0.5].
    """
    return float(arms ** 1.5 * np.sqrt(duration))


def significant_duration(arias: np.ndarray, dt: float, lower: float = 0.05,
                         upper: float = 0.95) -> Tuple[float, float, float]:
    """Significant duration between two fractions of the Arias intensity.

    References
    ----------
    Trifunac, M. D., and Brady, A. G. (1975). A study on the duration of
    strong earthquake ground motion. Bulletin of the Seismological Society of
    America, 65(3), 581-626.

    Parameters
    ----------
    arias : numpy.ndarray
        Cumulative Arias intensity from :func:`arias_series`.
    dt : float
        Time step [s].
    lower, upper : float, optional
        Fractions of the total, 0.05 and 0.95 by default.

    Returns
    -------
    start, end, duration : float
        Both instants and their difference [s], unrounded.
    """
    arias = np.asarray(arias, dtype=float)
    total = arias[-1]
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan"), float("nan")
    inside = np.flatnonzero((arias >= lower * total)
                            & (arias <= upper * total))
    if inside.size == 0:
        return float("nan"), float("nan"), float("nan")
    start = float(inside[0] * dt)
    end = float(inside[-1] * dt)
    return start, end, end - start


def bracketed_duration(acc: np.ndarray, dt: float,
                       threshold: float = 0.05) -> float:
    """Time between the first and last excursion beyond a threshold.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    threshold : float, optional
        Amplitude threshold [g], 0.05 by default.

    Returns
    -------
    float
        Bracketed duration [s]; 0.0 if the threshold is never reached.
    """
    acc = np.asarray(acc)
    inside = np.flatnonzero(np.abs(acc) >= threshold)
    if inside.size == 0:
        return 0.0
    return float((inside[-1] - inside[0]) * dt)


def bracketed_interval(acc: np.ndarray, dt: float,
                       threshold: float = 0.05) -> Tuple[float, float]:
    """First and last instants beyond a threshold.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    threshold : float, optional
        Amplitude threshold [g], 0.05 by default.

    Returns
    -------
    start, end : float
        Instants [s], both NaN if the threshold is never reached.
    """
    acc = np.asarray(acc)
    inside = np.flatnonzero(np.abs(acc) >= threshold)
    if inside.size == 0:
        return float("nan"), float("nan")
    return float(inside[0] * dt), float(inside[-1] * dt)


def uniform_duration(acc: np.ndarray, dt: float,
                     threshold: float = 0.05) -> float:
    """Total time spent beyond a threshold.

    The instants themselves are ``time[abs(acc) >= threshold]``.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    threshold : float, optional
        Amplitude threshold [g], 0.05 by default.

    Returns
    -------
    float
        Uniform duration [s].
    """
    acc = np.asarray(acc)
    return float(np.count_nonzero(np.abs(acc) >= threshold) * dt)


def housner_intensity(psv: np.ndarray, periods: np.ndarray,
                      lower: float = 0.1, upper: float = 2.5) -> float:
    """Housner (velocity spectrum) intensity.

    References
    ----------
    Housner, G. W. (1952). Spectrum intensities of strong-motion earthquakes.
    Proceedings of the Symposium on Earthquake and Blast Effects on
    Structures, EERI, 20-36.

    Parameters
    ----------
    psv : numpy.ndarray
        Pseudo-spectral velocity [cm/s].
    periods : numpy.ndarray
        Periods [s], spanning ``[lower, upper]``.
    lower, upper : float, optional
        Integration range [s], 0.1 to 2.5 by default.

    Returns
    -------
    float
        Housner intensity [cm], or NaN if the range is not spanned.
    """
    return _integrate_range(psv, periods, lower, upper)


def acceleration_spectrum_intensity(psa: np.ndarray, periods: np.ndarray,
                                    lower: float = 0.1,
                                    upper: float = 0.5) -> float:
    """Acceleration spectrum intensity.

    References
    ----------
    Von Thun, J. L., Roehm, L. H., Scott, G. A., and Wilson, J. A. (1988).
    Earthquake ground motions for design and analysis of dams. Earthquake
    Engineering and Soil Dynamics II, ASCE, 463-481.

    Parameters
    ----------
    psa : numpy.ndarray
        Pseudo-spectral acceleration [g].
    periods : numpy.ndarray
        Periods [s], spanning ``[lower, upper]``.
    lower, upper : float, optional
        Integration range [s], 0.1 to 0.5 by default.

    Returns
    -------
    float
        ASI [g.s], or NaN if the range is not spanned.
    """
    return _integrate_range(psa, periods, lower, upper)


def modified_acceleration_spectrum_intensity(
        psa: np.ndarray, periods: np.ndarray, lower: float = 0.1,
        upper: float = 2.5) -> float:
    """Modified acceleration spectrum intensity.

    Parameters
    ----------
    psa : numpy.ndarray
        Pseudo-spectral acceleration [g].
    periods : numpy.ndarray
        Periods [s], spanning ``[lower, upper]``.
    lower, upper : float, optional
        Integration range [s], 0.1 to 2.5 by default.

    Returns
    -------
    float
        MASI [g.s], or NaN if the range is not spanned.
    """
    return _integrate_range(psa, periods, lower, upper)


def displacement_spectrum_intensity(sd: np.ndarray, periods: np.ndarray,
                                    lower: float = 2.0,
                                    upper: float = 5.0) -> float:
    """Displacement spectrum intensity.

    References
    ----------
    Bradley, B. A. (2011). Empirical equations for the prediction of
    displacement spectrum intensity and its correlation with other intensity
    measures. Soil Dynamics and Earthquake Engineering, 31(8), 1182-1191.

    Parameters
    ----------
    sd : numpy.ndarray
        Spectral displacement [cm].
    periods : numpy.ndarray
        Periods [s], spanning ``[lower, upper]``.
    lower, upper : float, optional
        Integration range [s], 2.0 to 5.0 by default.

    Returns
    -------
    float
        DSI [cm.s], or NaN if the range is not spanned.
    """
    return _integrate_range(sd, periods, lower, upper)


def fourier_amplitude_spectrum(acc: np.ndarray, dt: float,
                               dtype=np.float32
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided Fourier amplitude spectrum.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    dtype : numpy.dtype, optional
        Float type of the result, float32 by default.

    Returns
    -------
    frequency : numpy.ndarray
        Positive frequencies [Hz].
    amplitude : numpy.ndarray
        Fourier amplitude [g.s].
    """
    acc = np.asarray(acc, dtype=dtype)
    padded = 2 ** int(np.ceil(np.log2(acc.size)))
    amplitude = np.abs(fftshift(fft(acc, padded))) * dt
    frequency = fftshift(fftfreq(padded, dt))
    positive = frequency > 0
    return (frequency[positive].astype(dtype),
            amplitude[positive].astype(dtype))


def power_amplitude_spectrum(acc: np.ndarray, dt: float,
                             dtype=np.float32
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised power amplitude spectrum.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    dtype : numpy.dtype, optional
        Float type of the result, float32 by default.

    Returns
    -------
    frequency : numpy.ndarray
        Positive frequencies [Hz].
    power : numpy.ndarray
        Power amplitude, normalised by the record duration and the mean square
        acceleration, so it is dimensionless.
    """
    frequency, amplitude = fourier_amplitude_spectrum(acc, dt, dtype)
    acc = np.asarray(acc, dtype=np.float64)
    duration = dt * (acc.size - 1)
    arms = root_mean_square(acc, dt)
    if duration <= 0 or not np.isfinite(arms) or arms == 0:
        return frequency, np.full(frequency.size, np.nan, dtype=dtype)
    power = amplitude.astype(np.float64) ** 2 / (np.pi * duration * arms ** 2)
    return frequency, power.astype(dtype)


def mean_period(frequency: np.ndarray, amplitude: np.ndarray,
                lower: float = 0.25, upper: float = 20.0) -> float:
    """Mean period.

    References
    ----------
    Rathje, E. M., Abrahamson, N. A., and Bray, J. D. (1998). Simplified
    frequency content estimates of earthquake ground motions. Journal of
    Geotechnical and Geoenvironmental Engineering, 124(2), 150-159.

    Parameters
    ----------
    frequency, amplitude : numpy.ndarray
        Fourier amplitude spectrum from :func:`fourier_amplitude_spectrum`.
    lower, upper : float, optional
        Frequency band [Hz], 0.25 to 20 by default.

    Returns
    -------
    float
        Mean period [s].
    """
    frequency = np.asarray(frequency, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    inside = (frequency > lower) & (frequency < upper)
    if not inside.any():
        return float("nan")
    weights = amplitude[inside] ** 2
    return float(np.sum(weights / frequency[inside]) / np.sum(weights))


def predominant_period(psa: np.ndarray, periods: np.ndarray) -> float:
    """Period at which the response spectrum peaks.

    Parameters
    ----------
    psa : numpy.ndarray
        Pseudo-spectral acceleration [g].
    periods : numpy.ndarray
        Periods [s].

    Returns
    -------
    float
        Predominant period [s].
    """
    psa = np.asarray(psa)
    periods = np.asarray(periods, dtype=float)
    return float(periods[int(np.argmax(psa))])


def filtered_incremental_velocity(
        acc: np.ndarray, dt: float, periods: np.ndarray, alpha: float = 0.7,
        beta: float = None,
        filtering: Literal["causal", "acausal"] = "causal",
        dtype=np.float32) -> np.ndarray:
    """Filtered incremental velocity, FIV3.

    The trace is low-pass filtered at ``beta / Tn`` and integrated over a
    sliding window of ``alpha * Tn``. FIV3 is the larger of the sums of the
    three largest peaks and of the three largest troughs.

    References
    ----------
    Davalos, H., and Miranda, E. (2019). Filtered incremental velocity: A
    novel approach in intensity measures for seismic collapse estimation.
    Earthquake Engineering & Structural Dynamics, 48(12), 1384-1405.

    Dávalos H., Heresi P., and Miranda E. (2020). A ground motion prediction
    equation for filtered incremental velocity, FIV3. Soil Dynamics and
    Earthquake Engineering, 139, 106346.
    https://doi.org/10.1016/j.soildyn.2020.106346.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration [g].
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Periods [s].
    alpha : float, optional
        Integration window as a multiple of the period, 0.7 by default.
    beta : float, optional
        Low-pass cut-off as ``beta / Tn`` [Hz], None by default.
        This collapses cut-off frequency to a fixed 1 Hz at every period as
        suggested in Dávalos et al. 2020.
    filtering : {'causal', 'acausal'}, optional
        One-pass or zero-phase filtering, 'causal' by default.
    dtype : numpy.dtype, optional
        Float type of the result, float32 by default.

    Returns
    -------
    numpy.ndarray
        FIV3 at each period [cm/s].
    """
    acc = np.asarray(acc, dtype=dtype)
    periods = np.atleast_1d(np.asarray(periods, dtype=dtype))
    nyquist = 0.5 / dt

    out = np.empty(periods.size, dtype=dtype)
    for index, period in enumerate(periods):
        cut_off = (period if beta is None else beta) / period
        numerator, denominator = butter(2, cut_off / nyquist, "lowpass")
        if filtering == "causal":
            filtered = lfilter(numerator, denominator, acc)
        elif filtering == "acausal":
            filtered = filtfilt(numerator, denominator, acc)
        else:
            raise ValueError("filtering must be 'causal' or 'acausal'")

        window = int(np.floor(alpha * period / dt)) + 1
        count = filtered.size - window + 1
        if count < 3:
            out[index] = np.nan
            continue

        running = np.concatenate(
            ([0.0], np.cumsum(filtered, dtype=np.float64)))
        totals = running[window:window + count] - running[:count]
        ends = filtered[:count] + filtered[window - 1:window - 1 + count]
        incremental = dt * (totals - 0.5 * ends)

        peaks, _ = find_peaks(incremental)
        troughs, _ = find_peaks(-incremental)
        top = np.sort(incremental[peaks])[-3:]
        bottom = np.abs(np.sort(incremental[troughs])[:3])
        out[index] = max(top.sum(), bottom.sum())

    return out * np.dtype(dtype).type(G_TO_CM)


def rotated_filtered_incremental_velocity(
        acc1: np.ndarray, acc2: np.ndarray, dt: float, periods: np.ndarray,
        alpha: float = 0.7, beta: Optional[float] = None,
        filtering: Literal["causal", "acausal"] = "causal",
        angles: Optional[np.ndarray] = None,
        dtype=np.float32) -> np.ndarray:
    """:func:`filtered_incremental_velocity` at every rotation angle.

    Filtering and windowed integration are linear, so the rotated
    incremental velocity is a combination of the two components' own. Each
    component is filtered once per period, and only the peak search runs per
    angle.

    References
    ----------
    Davalos, H., and Miranda, E. (2019). Filtered incremental velocity: A
    novel approach in intensity measures for seismic collapse estimation.
    Journal of Earthquake Engineering, 23(9), 1384-1405.

    Parameters
    ----------
    acc1, acc2 : numpy.ndarray
        Horizontal acceleration traces [g]; the shorter is zero-padded.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Periods [s].
    alpha, beta, filtering
        See :func:`filtered_incremental_velocity`.
    angles : numpy.ndarray, optional
        Rotation angles [deg], :data:`DEFAULT_ANGLES` if omitted.
    dtype : numpy.dtype, optional
        Float type of the result, float32 by default.

    Returns
    -------
    numpy.ndarray
        FIV3 [cm/s], shape ``(len(periods), len(angles))``.
    """
    acc1 = np.asarray(acc1, dtype=dtype)
    acc2 = np.asarray(acc2, dtype=dtype)
    if acc1.size != acc2.size:
        width = max(acc1.size, acc2.size)
        acc1 = np.pad(acc1, (0, width - acc1.size))
        acc2 = np.pad(acc2, (0, width - acc2.size))

    periods = _as_periods(periods, dtype)
    angles = DEFAULT_ANGLES if angles is None else np.atleast_1d(
        np.asarray(angles, dtype=dtype))
    radians = np.deg2rad(angles)
    cosines, sines = np.cos(radians), np.sin(radians)
    nyquist = 0.5 / dt

    def increments(trace, window, count):
        """The windowed incremental velocity of one filtered trace."""
        running = np.concatenate(([0.0], np.cumsum(trace, dtype=np.float64)))
        totals = running[window:window + count] - running[:count]
        ends = trace[:count] + trace[window - 1:window - 1 + count]
        return dt * (totals - 0.5 * ends)

    out = np.empty((periods.size, angles.size), dtype=dtype)
    for index, period in enumerate(periods):
        cut_off = (period if beta is None else beta) / period
        numerator, denominator = butter(2, cut_off / nyquist, "lowpass")
        if filtering == "causal":
            first = lfilter(numerator, denominator, acc1)
            second = lfilter(numerator, denominator, acc2)
        elif filtering == "acausal":
            first = filtfilt(numerator, denominator, acc1)
            second = filtfilt(numerator, denominator, acc2)
        else:
            raise ValueError("filtering must be 'causal' or 'acausal'")

        window = int(np.floor(alpha * period / dt)) + 1
        count = first.size - window + 1
        if count < 3:
            out[index] = np.nan
            continue

        one = increments(first, window, count)
        two = increments(second, window, count)
        for column in range(angles.size):
            incremental = one * cosines[column] + two * sines[column]
            peaks, _ = find_peaks(incremental)
            troughs, _ = find_peaks(-incremental)
            top = np.sort(incremental[peaks])[-3:]
            bottom = np.abs(np.sort(incremental[troughs])[:3])
            out[index, column] = max(top.sum(), bottom.sum())

    return out * np.dtype(dtype).type(G_TO_CM)


def _dtype(precision) -> np.dtype:
    """The float type for a precision, which may already be a dtype."""
    if precision in PRECISIONS:
        return np.dtype(PRECISIONS[precision])
    return np.dtype(precision)


def _as_periods(periods, dtype=np.float32) -> np.ndarray:
    """Coerce a period argument to a 1-D float array of ``dtype``."""
    if isinstance(periods, (int, float)):
        periods = [periods]
    values = np.asarray(periods, dtype=dtype).ravel()
    if values.size == 0:
        raise ValueError("no periods requested")
    if np.any(values <= 0):
        raise ValueError("periods must be positive; got a minimum of %g"
                         % values.min())
    return values


def resampling_factors(dt: float, periods: np.ndarray,
                       min_points=UNSET) -> np.ndarray:
    """Upsampling factor per period, so that ``dt <= T / min_points``.

    Parameters
    ----------
    dt : float
        Time step of the record [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    min_points : int, optional
        Samples per period; :data:`MIN_POINTS_PER_PERIOD` if omitted, and
        ``None`` or 0 disables resampling.

    Returns
    -------
    numpy.ndarray
        One integer factor per period, at least 1.
    """
    periods = _as_periods(periods)
    if min_points is UNSET:
        min_points = MIN_POINTS_PER_PERIOD
    if not min_points:
        return np.ones(periods.size, dtype=int)
    return np.maximum(1, np.ceil(dt * min_points / periods)).astype(int)


def resample(acc: np.ndarray, dt: float, factor: int,
             dtype=np.float32) -> Tuple[np.ndarray, float]:
    """Linearly interpolate a trace onto a step of ``dt / factor``.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration trace.
    dt : float
        Time step [s].
    factor : int
        Upsampling factor; 1 returns the trace unchanged.

    Returns
    -------
    trace : numpy.ndarray
        The interpolated trace.
    step : float
        Its time step [s].
    """
    acc = np.asarray(acc, dtype=dtype)
    if factor <= 1:
        return acc, dt
    end = dt * (acc.size - 1)
    # linspace pins both ends exactly; arange(n) * step can drift past the
    # last sample and leave the interpolation out of range
    count = int(round(end / (dt / factor))) + 1
    fine = np.linspace(0.0, end, count)
    coarse = np.linspace(0.0, end, acc.size)
    return np.interp(fine, coarse, acc).astype(dtype), fine[1] - fine[0]


def _groups(acc, dt, periods, min_points, dtype=np.float32):
    """Yield ``(indices, trace, step)`` per distinct resampling factor."""
    factors = resampling_factors(dt, periods, min_points)
    for factor in np.unique(factors):
        indices = np.flatnonzero(factors == factor)
        trace, step = resample(acc, dt, int(factor), dtype)
        yield indices, trace, step


def response(acc: np.ndarray, dt: float, periods: np.ndarray,
             damping: float = 0.05,
             solver: Optional[Literal["newmark", "exact"]] = None,
             derivatives: int = 0, precision: int = 32
             ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Relative response history of the oscillators.

    Displacement, velocity and acceleration relative to the ground; only the
    ``derivatives`` asked for are returned.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration trace.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    damping : float, optional
        Damping ratio, 0.05 by default.
    solver : {'newmark', 'exact'}, optional
        :data:`DEFAULT_SOLVER` if omitted.
    derivatives : int, optional
        How many time derivatives to return alongside the displacement: 0 for
        none, 1 for the velocity, 2 for the velocity and the acceleration.
    precision : int, optional
        32 or 64; see :data:`PRECISIONS`.

    Returns
    -------
    u : numpy.ndarray
        Relative displacement, shape ``(len(acc), len(periods))``.
    v, a : numpy.ndarray
        Relative velocity and acceleration, returned as ``derivatives`` asks.
    """
    solver = DEFAULT_SOLVER if solver is None else solver
    if solver == "newmark":
        u, u_dot, u_ddot = lrha(acc, dt, periods, damping,
                                precision=precision)
    elif solver == "exact":
        u, u_dot, u_ddot = nigam_jennings(acc, dt, periods, damping,
                                          precision=precision)
    else:
        raise ValueError("solver must be one of %s; got %r"
                         % (SOLVERS, solver))
    if derivatives == 0:
        return u
    if derivatives == 1:
        return u, u_dot
    if derivatives == 2:
        return u, u_dot, u_ddot
    raise ValueError("derivatives must be 0, 1 or 2; got %r" % derivatives)


def response_maxima(acc: np.ndarray, dt: float, periods: np.ndarray,
                    damping: float = 0.05, chunk: Optional[int] = None,
                    solver: Optional[Literal["newmark", "exact"]] = None,
                    min_points=UNSET,
                    precision: int = 32) -> dict:
    """Every peak response of the oscillator, and the input energy spectra.

    The absolute responses add the ground motion to the relative ones, e.g.
    ``a_abs = a_rel + ag``. Input energy per unit mass follows Uang and
    Bertero (1990): ``integral(-ag du)`` (relative) and
    ``integral(a_abs du_g)`` (absolute), both integrated along the
    displacement path.

    References
    ----------
    Uang, C. M., and Bertero, V. V. (1990). Evaluation of seismic energy in
    structures. Earthquake Engineering & Structural Dynamics, 19(1), 77-90.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration trace.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    damping : float, optional
        Damping ratio, 0.05 by default.
    chunk : int, optional
        Periods solved at once; sized from :data:`MEMORY_BUDGET` if omitted.
    solver : {'newmark', 'exact'}, optional
        :data:`DEFAULT_SOLVER` if omitted.

    Returns
    -------
    dict
        ``sd``, ``sv``, ``sa_rel``, ``sa_abs``, ``sv_abs``, ``sd_abs``,
        ``ei_rel`` and ``ei_abs``, one array per key. Displacements are in the
        acceleration's unit times s2, velocities times s, accelerations in the
        acceleration's own unit, and the energies in its unit squared times
        s2.
    """
    dtype = _dtype(precision)
    acc = np.asarray(acc, dtype=dtype)
    periods = _as_periods(periods, dtype)
    damping = dtype.type(damping)

    keys = ("sd_rel", "sv_rel", "sa_rel", "sa_abs", "sv_abs",
            "sd_abs", "ei_rel",
            "ei_abs")
    out = {key: np.empty(periods.size, dtype=dtype) for key in keys}

    for indices, trace, step in _groups(acc, dt, periods, min_points,
                                        dtype):
        vel = cumulative_trapezoid(trace, dx=step,
                                   initial=0.0).astype(dtype)
        disp = cumulative_trapezoid(vel, dx=step,
                                    initial=0.0).astype(dtype)
        block = chunk or chunk_length(trace.size, 7)
        for start in range(0, indices.size, block):
            here = indices[start:start + block]
            ur, vr, ar = response(trace, step, periods[here], damping,
                                  solver, derivatives=2,
                                  precision=precision)

            a_tot = ar + trace[:, None]
            out["sd_rel"][here] = np.max(np.abs(ur), axis=0)
            out["sv_rel"][here] = np.max(np.abs(vr), axis=0)
            out["sa_rel"][here] = np.max(np.abs(ar), axis=0)
            out["sa_abs"][here] = np.max(np.abs(a_tot), axis=0)
            out["sv_abs"][here] = np.max(np.abs(vr + vel[:, None]), axis=0)
            out["sd_abs"][here] = np.max(np.abs(ur + disp[:, None]), axis=0)
            # broadcast views, so nothing of shape (npts, n_chunk) is copied
            out["ei_rel"][here] = trapezoid(
                np.broadcast_to(-trace[:, None], ur.shape), ur, axis=0)
            out["ei_abs"][here] = trapezoid(
                a_tot, np.broadcast_to(disp[:, None], ur.shape), axis=0)
            del ur, vr, ar, a_tot
    return out


def chunk_length(npts: int, per_period_arrays: int,
                 budget: int = MEMORY_BUDGET) -> int:
    """How many periods to process at once inside a memory budget.

    Parameters
    ----------
    npts : int
        Length of the acceleration trace.
    per_period_arrays : int
        Number of ``(npts, n_chunk)`` float32 arrays live at once.
    budget : int, optional
        Bytes to stay within, :data:`MEMORY_BUDGET` by default.

    Returns
    -------
    int
        At least 1.
    """
    per_period = max(1, npts * 4 * per_period_arrays)
    return max(1, int(budget // per_period))


def spectral_displacement(acc: np.ndarray, dt: float, periods: np.ndarray,
                          damping: float = 0.05,
                          chunk: Optional[int] = None,
                          solver: Optional[Literal["newmark", "exact"]] = None,
                          min_points=UNSET,
                          precision: int = 32) -> np.ndarray:
    """Peak relative displacement at each period.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration trace.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    damping : float, optional
        Damping ratio, 0.05 by default.
    chunk : int, optional
        Periods solved at once; sized from :data:`MEMORY_BUDGET` if omitted.
    solver : {'newmark', 'exact'}, optional
        :data:`DEFAULT_SOLVER` if omitted.
    min_points : int, optional
        Samples per oscillator period; the record is interpolated where it is
        coarser. :data:`MIN_POINTS_PER_PERIOD` if omitted, ``None`` to solve
        at the record's own step.

    Returns
    -------
    numpy.ndarray
        Sd at each period, in the acceleration's unit times s2.
    """
    dtype = _dtype(precision)
    acc = np.asarray(acc, dtype=dtype)
    periods = _as_periods(periods, dtype)
    damping = dtype.type(damping)

    sd = np.empty(periods.size, dtype=dtype)
    for indices, trace, step in _groups(acc, dt, periods, min_points,
                                        dtype):
        block = chunk or chunk_length(trace.size, 5)
        for start in range(0, indices.size, block):
            here = indices[start:start + block]
            u = response(trace, step, periods[here], damping, solver,
                         precision=precision)
            sd[here] = np.max(np.abs(u), axis=0)
    return sd


def _support_function(u1: np.ndarray, u2: np.ndarray,
                      directions: np.ndarray) -> np.ndarray:
    """``max over t of |u1 dx + u2 dy|`` for every direction.

    The rotated peak is the support function of the convex hull of the
    response path, since a linear functional attains its maximum at a hull
    vertex. This is exact, and only the hull vertices are searched.

    Parameters
    ----------
    u1, u2 : numpy.ndarray
        Response histories of the two components at one period.
    directions : numpy.ndarray
        Shape ``(2, n_angles)``: cosines and sines of the angles.

    Returns
    -------
    numpy.ndarray
        One peak per direction.
    """
    path = np.column_stack((u1, u2)).astype(np.float64)
    try:
        vertices = path[ConvexHull(path).vertices]
    except (QhullError, ValueError):
        vertices = path            # degenerate: hull is not two-dimensional
    return np.max(np.abs(vertices @ directions), axis=0)


def rotated_spectral_displacement(
        acc1: np.ndarray, acc2: np.ndarray, dt: float, periods: np.ndarray,
        damping: float = 0.05, angles: Optional[np.ndarray] = None,
        chunk: Optional[int] = None,
        method: Literal["hull", "angles"] = "hull",
        solver: Optional[Literal["newmark", "exact"]] = None,
        min_points=UNSET,
        precision: int = 32) -> np.ndarray:
    """Peak relative displacement at every period and rotation angle.

    One solve serves every RotDxx percentile and both component spectra.

    References
    ----------
    Boore, D. M. (2010). Orientation-independent, nongeometric-mean measures
    of seismic intensity from two horizontal components of motion. Bulletin of
    the Seismological Society of America, 100(4), 1830-1835.

    Parameters
    ----------
    acc1, acc2 : numpy.ndarray
        Horizontal acceleration traces; the shorter is zero-padded.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    damping : float, optional
        Damping ratio, 0.05 by default.
    angles : numpy.ndarray, optional
        Rotation angles [deg], :data:`DEFAULT_ANGLES` if omitted.
    chunk : int, optional
        Periods solved at once; sized from :data:`MEMORY_BUDGET` if omitted.
    method : {'hull', 'angles'}, optional
        'hull' uses :func:`_support_function`; 'angles' reduces the rotated
        history at each angle and is kept as a reference implementation.
    solver : {'newmark', 'exact'}, optional
        :data:`DEFAULT_SOLVER` if omitted.
    min_points : int, optional
        Samples per oscillator period; the record is interpolated where it is
        coarser. :data:`MIN_POINTS_PER_PERIOD` if omitted, ``None`` to solve
        at the record's own step.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(periods), len(angles))``. With the default angles,
        columns 0 and 90 are the two components' own spectra exactly.
    """
    dtype = _dtype(precision)
    acc1 = np.asarray(acc1, dtype=dtype)
    acc2 = np.asarray(acc2, dtype=dtype)
    if acc1.size != acc2.size:
        width = max(acc1.size, acc2.size)
        acc1 = np.pad(acc1, (0, width - acc1.size))
        acc2 = np.pad(acc2, (0, width - acc2.size))

    periods = _as_periods(periods, dtype)
    damping = dtype.type(damping)
    angles = DEFAULT_ANGLES if angles is None else np.asarray(
        angles, dtype=dtype)
    if method not in ("hull", "angles"):
        raise ValueError("method must be 'hull' or 'angles'; got %r" % method)

    radians = np.deg2rad(angles)
    directions = np.vstack((np.cos(radians), np.sin(radians)))
    single = directions.astype(dtype)

    sd = np.empty((periods.size, angles.size), dtype=dtype)
    factors = resampling_factors(dt, periods, min_points)
    for factor in np.unique(factors):
        indices = np.flatnonzero(factors == factor)
        trace1, step = resample(acc1, dt, int(factor), dtype)
        trace2, _ = resample(acc2, dt, int(factor), dtype)
        block = chunk or chunk_length(trace1.size, 9)
        for start in range(0, indices.size, block):
            here = indices[start:start + block]
            u1 = response(trace1, step, periods[here], damping, solver,
                          precision=precision)
            u2 = response(trace2, step, periods[here], damping, solver,
                          precision=precision)

            if method == "hull":
                for column, index in enumerate(here):
                    sd[index] = _support_function(
                        u1[:, column], u2[:, column], directions)
            else:
                rotated = np.empty_like(u1)
                for k in range(angles.size):
                    np.multiply(u1, single[0, k], out=rotated)
                    rotated += u2 * single[1, k]
                    np.abs(rotated, out=rotated)
                    sd[here, k] = rotated.max(axis=0)
                del rotated
            del u1, u2
    return sd


def rotated_response_maxima(
        acc1: np.ndarray, acc2: np.ndarray, dt: float, periods: np.ndarray,
        damping: float = 0.05, angles: Optional[np.ndarray] = None,
        chunk: Optional[int] = None,
        solver: Optional[Literal["newmark", "exact"]] = None,
        min_points=UNSET,
        precision: int = 32) -> dict:
    """:func:`response_maxima` at every rotation angle.

    The six peaks come from one convex hull per period
    (:func:`_support_function`). The two energies are bilinear in the
    components, so each is a quadratic form in ``(cos, sin)`` and four path
    integrals per period give every angle exactly.

    References
    ----------
    Boore, D. M. (2010). Orientation-independent, nongeometric-mean measures
    of seismic intensity from two horizontal components. Bulletin of the
    Seismological Society of America, 100(4), 1830-1835.

    Uang, C. M., and Bertero, V. V. (1990). Evaluation of seismic energy in
    structures. Earthquake Engineering & Structural Dynamics, 19(1), 77-90.

    Parameters
    ----------
    acc1, acc2 : numpy.ndarray
        Horizontal acceleration traces; the shorter is zero-padded.
    dt : float
        Time step [s].
    periods : numpy.ndarray
        Oscillator periods [s].
    damping : float, optional
        Damping ratio, 0.05 by default.
    angles : numpy.ndarray, optional
        Rotation angles [deg], :data:`DEFAULT_ANGLES` if omitted.
    chunk : int, optional
        Periods solved at once; sized from :data:`MEMORY_BUDGET` if omitted.
    solver : {'newmark', 'exact'}, optional
        :data:`DEFAULT_SOLVER` if omitted.
    min_points : int, optional
        Samples per oscillator period; the record is interpolated where it is
        coarser. :data:`MIN_POINTS_PER_PERIOD` if omitted, ``None`` to solve
        at the record's own step.

    Returns
    -------
    dict
        The same keys as :func:`response_maxima`, each of shape
        ``(len(periods), len(angles))``.
    """
    dtype = _dtype(precision)
    acc1 = np.asarray(acc1, dtype=dtype)
    acc2 = np.asarray(acc2, dtype=dtype)
    if acc1.size != acc2.size:
        width = max(acc1.size, acc2.size)
        acc1 = np.pad(acc1, (0, width - acc1.size))
        acc2 = np.pad(acc2, (0, width - acc2.size))

    periods = _as_periods(periods, dtype)
    damping = dtype.type(damping)
    angles = DEFAULT_ANGLES if angles is None else np.asarray(
        angles, dtype=dtype)

    radians = np.deg2rad(angles)
    cosines, sines = np.cos(radians), np.sin(radians)
    directions = np.vstack((cosines, sines))

    keys = ("sd_rel", "sv_rel", "sa_rel", "sa_abs", "sv_abs",
            "sd_abs", "ei_rel",
            "ei_abs")
    out = {key: np.empty((periods.size, angles.size), dtype=dtype)
           for key in keys}

    def integrate(drive, path):
        """Path integral of ``drive`` along ``path``, accumulated in float64
        so the cross terms do not lose the cancellation."""
        if drive.ndim == 1:
            drive = np.broadcast_to(drive[:, None], path.shape)
        return trapezoid(np.asarray(drive, dtype=np.float64),
                         np.asarray(path, dtype=np.float64), axis=0)

    factors = resampling_factors(dt, periods, min_points)
    for factor in np.unique(factors):
        indices = np.flatnonzero(factors == factor)
        trace1, step = resample(acc1, dt, int(factor), dtype)
        trace2, _ = resample(acc2, dt, int(factor), dtype)

        ground = []
        for trace in (trace1, trace2):
            velocity = cumulative_trapezoid(trace, dx=step,
                                            initial=0.0).astype(dtype)
            ground.append((velocity,
                           cumulative_trapezoid(velocity, dx=step,
                                                initial=0.0
                                                ).astype(dtype)))

        block = chunk or chunk_length(trace1.size, 18)
        for start in range(0, indices.size, block):
            here = indices[start:start + block]
            u1, v1, relative1 = response(trace1, step, periods[here],
                                         damping, solver, derivatives=2,
                                         precision=precision)
            u2, v2, relative2 = response(trace2, step, periods[here],
                                         damping, solver, derivatives=2,
                                         precision=precision)

            absolute1 = relative1 + trace1[:, None]
            absolute2 = relative2 + trace2[:, None]
            pairs = {
                "sd_rel": (u1, u2),
                "sv_rel": (v1, v2),
                "sa_rel": (relative1, relative2),
                "sa_abs": (absolute1, absolute2),
                "sv_abs": (v1 + ground[0][0][:, None],
                           v2 + ground[1][0][:, None]),
                "sd_abs": (u1 + ground[0][1][:, None],
                           u2 + ground[1][1][:, None]),
            }
            for key, (first, second) in pairs.items():
                for column, index in enumerate(here):
                    out[key][index] = _support_function(
                        first[:, column], second[:, column], directions)

            # the energies are quadratic forms; four cross integrals
            for key, (drive1, drive2, path1, path2) in (
                    ("ei_rel", (-trace1, -trace2, u1, u2)),
                    ("ei_abs", (absolute1, absolute2,
                                np.broadcast_to(ground[0][1][:, None],
                                                u1.shape),
                                np.broadcast_to(ground[1][1][:, None],
                                                u2.shape)))):
                e11 = integrate(drive1, path1)
                e12 = integrate(drive1, path2)
                e21 = integrate(drive2, path1)
                e22 = integrate(drive2, path2)
                out[key][here] = (
                    e11[:, None] * cosines ** 2
                    + (e12 + e21)[:, None] * cosines * sines
                    + e22[:, None] * sines ** 2)

            del u1, v1, relative1, u2, v2, relative2
            del absolute1, absolute2, pairs
    return out


def pseudo_spectra(sd: np.ndarray,
                   periods: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pseudo-velocity and pseudo-acceleration from spectral displacement.

    Parameters
    ----------
    sd : numpy.ndarray
        Spectral displacement, shape ``(n_periods,)`` or
        ``(n_periods, n_angles)``.
    periods : numpy.ndarray
        Periods [s] indexing the first axis.

    Returns
    -------
    psv, psa : numpy.ndarray
        Same shape as ``sd``.
    """
    periods = _as_periods(periods)
    sd = np.asarray(sd, dtype=np.float32)
    omega = (2.0 * np.pi / periods).astype(np.float32)
    if sd.ndim == 2:
        omega = omega[:, np.newaxis]
    psv = sd * omega
    psa = psv * omega
    return psv, psa


def sa_avg_windows(periods: np.ndarray, t_low: float = 0.2,
                   t_high: float = 2.0, n_per: int = 10,
                   dtype=np.float32) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Averaging windows behind Sa_avg, and their union.

    References
    ----------
    Baker, J. W., and Cornell, C. A. (2006). Spectral shape, epsilon and
    record selection. Earthquake Engineering & Structural Dynamics, 35(9),
    1077-1095.

    Parameters
    ----------
    periods : numpy.ndarray
        Anchor periods T0 [s].
    t_low, t_high : float, optional
        Averaging bounds as multiples of T0, 0.2 and 2.0 by default.
    n_per : int, optional
        Periods per window, 10 by default.
    dtype : numpy.dtype, optional
        Float type of the periods, float32 by default.

    Returns
    -------
    unique : numpy.ndarray
        Every distinct period any window needs, sorted.
    indices : list of numpy.ndarray
        Positions in ``unique`` used by each anchor period's window.
    """
    periods = _as_periods(periods, dtype)
    windows = [np.linspace(t_low * anchor, t_high * anchor, num=n_per,
                           endpoint=True, dtype=dtype)
               for anchor in periods]
    unique = np.unique(np.concatenate(windows)).astype(dtype)
    lookup = {value: position for position, value in enumerate(unique)}
    indices = [np.array([lookup[value] for value in window], dtype=np.intp)
               for window in windows]
    return unique, indices


def average_over_windows(sa: np.ndarray, indices: List[np.ndarray],
                         dtype=np.float32) -> np.ndarray:
    """Geometric mean of ``sa`` over each window from :func:`sa_avg_windows`.

    Parameters
    ----------
    sa : numpy.ndarray
        Spectral acceleration over the unique periods, shape ``(n_unique,)``
        or ``(n_unique, n_angles)``.
    indices : list of numpy.ndarray
        Per-anchor index arrays from :func:`sa_avg_windows`.
    dtype : numpy.dtype, optional
        Float type of the result, float32 by default.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(indices),)`` or ``(len(indices), n_angles)``.
    """
    sa = np.asarray(sa, dtype=dtype)
    logs = np.log(sa)
    averaged = [np.exp(logs[window].mean(axis=0)) for window in indices]
    return np.asarray(averaged, dtype=dtype)
