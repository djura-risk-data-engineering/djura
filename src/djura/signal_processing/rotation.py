# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""RotDxx, generalised to any intensity measure.

A quantity is rotated by rotating the pair of horizontal series it is computed
from, ``x(t, theta) = x1 cos(theta) + x2 sin(theta)``, and RotDxx is the xx-th
percentile over theta. Which series is rotated differs by measure: PGA, PGV
and PGD are integrated first and rotated after, spectral measures rotate the
SDOF response, and anything else is recomputed at each angle by
:func:`rotated_scalar`.

References
----------
Boore, D. M. (2010). Orientation-independent, nongeometric-mean measures of
seismic intensity from two horizontal components of motion. Bulletin of the
Seismological Society of America, 100(4), 1830-1835.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .ims import DEFAULT_ANGLES, displacement, velocity


def _angles(angles, dtype=np.float32) -> np.ndarray:
    """Coerce the angle argument to degrees in ``dtype``."""
    if angles is None:
        return np.asarray(DEFAULT_ANGLES, dtype=dtype)
    return np.atleast_1d(np.asarray(angles, dtype=dtype))


def _pad_to_common(series1: np.ndarray, series2: np.ndarray,
                   dtype=None) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter of two traces to the longer one's length.

    ``dtype`` defaults to whichever of the two is the wider float type, so a
    float64 trace is not silently narrowed.
    """
    if dtype is None:
        dtype = np.result_type(np.asarray(series1).dtype,
                               np.asarray(series2).dtype, np.float32)
    series1 = np.asarray(series1, dtype=dtype)
    series2 = np.asarray(series2, dtype=dtype)
    if series1.size != series2.size:
        width = max(series1.size, series2.size)
        series1 = np.pad(series1, (0, width - series1.size))
        series2 = np.pad(series2, (0, width - series2.size))
    return series1, series2


def rotate(series1: np.ndarray, series2: np.ndarray,
           angles: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate a pair of series onto every angle.

    Parameters
    ----------
    series1, series2 : numpy.ndarray
        The two as-recorded horizontal series.
    angles : numpy.ndarray, optional
        Angles [deg], :data:`djura.signal_processing.ims.DEFAULT_ANGLES` if
        omitted.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(series1), len(angles))``.
    """
    series1, series2 = _pad_to_common(series1, series2)
    angles = _angles(angles, series1.dtype)
    radians = np.deg2rad(angles).astype(series1.dtype)
    return (series1[:, None] * np.cos(radians)
            + series2[:, None] * np.sin(radians))


def percentiles(by_angle: np.ndarray,
                xx: Union[float, Tuple[float, ...]]) -> np.ndarray:
    """Percentiles of a measure over the angle axis.

    Parameters
    ----------
    by_angle : numpy.ndarray
        Values at each angle, angle last.
    xx : float or sequence of float
        Percentiles, e.g. 50 for RotD50 or ``[50, 100]`` for both.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(xx), ...)`` for a sequence, or the leading shape alone
        for a scalar ``xx``.
    """
    by_angle = np.asarray(by_angle)
    dtype = (by_angle.dtype if by_angle.dtype.kind == "f" else np.float32)
    scalar = np.isscalar(xx)
    values = np.percentile(by_angle, np.atleast_1d(xx), axis=-1)
    values = values.astype(dtype)
    return values[0] if scalar else values


def rotated_peaks(acc1: np.ndarray, acc2: np.ndarray, dt: float,
                  xx: Union[float, Tuple[float, ...]] = (50, 100),
                  angles: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RotDxx peak ground acceleration, velocity and displacement.

    Integration is linear, so each component is integrated once and the three
    series are rotated, rather than re-integrating at every angle.

    Parameters
    ----------
    acc1, acc2 : numpy.ndarray
        Horizontal acceleration traces [g].
    dt : float
        Time step [s].
    xx : float or sequence of float, optional
        Percentiles, ``(50, 100)`` by default.
    angles : numpy.ndarray, optional
        Angles [deg]; the package default if omitted.

    Returns
    -------
    pga, pgv, pgd : numpy.ndarray
        In g, cm/s and cm, one value per requested percentile.
    """
    acc1, acc2 = _pad_to_common(acc1, acc2)
    vel1, vel2 = velocity(acc1, dt), velocity(acc2, dt)
    disp1, disp2 = displacement(vel1, dt), displacement(vel2, dt)

    out = []
    for first, second in ((acc1, acc2), (vel1, vel2), (disp1, disp2)):
        rotated = rotate(first, second, angles)
        out.append(percentiles(np.max(np.abs(rotated), axis=0), xx))
    return tuple(out)


def rotated_scalar(acc1: np.ndarray, acc2: np.ndarray, dt: float,
                   measure: Callable[[np.ndarray, float], object],
                   xx: Union[float, Tuple[float, ...]] = (50, 100),
                   angles: Optional[np.ndarray] = None) -> np.ndarray:
    """RotDxx of any measure computable from a single trace.

    Costs one evaluation of ``measure`` per angle.

    Parameters
    ----------
    acc1, acc2 : numpy.ndarray
        Horizontal acceleration traces [g].
    dt : float
        Time step [s].
    measure : callable
        Called as ``measure(acc, dt)``, returning a scalar or an array of the
        same shape at every angle.
    xx : float or sequence of float, optional
        Percentiles, ``(50, 100)`` by default.
    angles : numpy.ndarray, optional
        Angles [deg]; the package default if omitted.

    Returns
    -------
    numpy.ndarray
        The requested percentiles, taken at each position if ``measure``
        returns an array.

    Examples
    --------
    >>> from djura import signal_processing as sig
    >>> sig.rotation.rotated_scalar(a1, a2, dt, sig.ims.arias_intensity,
    ...                             xx=50)
    """
    acc1, acc2 = _pad_to_common(acc1, acc2)
    angles = _angles(angles, acc1.dtype)
    radians = np.deg2rad(angles).astype(acc1.dtype)

    values = []
    rotated = np.empty_like(acc1)
    for index in range(angles.size):
        np.multiply(acc1, np.cos(radians[index]), out=rotated)
        rotated += acc2 * np.sin(radians[index])
        values.append(measure(rotated, dt))
    stacked = np.asarray(values)
    return percentiles(np.moveaxis(stacked, 0, -1), xx)


def geometric_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Geometric mean of a measure over the two as-recorded components.

    Parameters
    ----------
    first, second : numpy.ndarray
        The measure on each component.

    Returns
    -------
    numpy.ndarray
        Geometric mean, in the same unit.
    """
    return np.sqrt(np.asarray(first, dtype=np.float64)
                   * np.asarray(second, dtype=np.float64)).astype(
                       np.result_type(np.asarray(first).dtype,
                                      np.asarray(second).dtype, np.float32))
