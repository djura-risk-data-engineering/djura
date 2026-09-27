# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Unit handling."""
import numpy as np

#: Gravity [m/s2]. See the module docstring on why this is not 9.80665.
GRAVITY = 9.81

#: Gravity [cm/s2]: takes a g-based quantity to a cm-based one.
G_TO_CM = 100.0 * GRAVITY

#: Accepted input units, and the factor taking a value in that unit to g.
ACCELERATION_UNITS = {
    "g": 1.0,
    "m/s2": 1.0 / GRAVITY,
    "m/s^2": 1.0 / GRAVITY,
    "m/s**2": 1.0 / GRAVITY,
    "cm/s2": 1.0 / G_TO_CM,
    "cm/s^2": 1.0 / G_TO_CM,
    "cm/s**2": 1.0 / G_TO_CM,
    "gal": 1.0 / G_TO_CM,
}

#: Unit of each intensity measure, keyed by its name on
#: :class:`~djura.signal_processing.gm.Component`.
IM_UNITS = {
    "pga": "g",
    "pgv": "cm/s",
    "pgd": "cm",
    "sa": "g",
    "psa": "g",
    "psv": "cm/s",
    "sd_rel": "cm",
    "sv_rel": "cm/s",
    "sa_rel": "g",
    "sa_abs": "g",
    "sv_abs": "cm/s",
    "sd_abs": "cm",
    "ei_rel": "m2/s2",
    "ei_abs": "m2/s2",
    "equivalent_velocity": "cm/s",
    "sa_avg": "g",
    "fiv3": "cm/s",
    "ia": "m/s",
    "cav": "g.s",
    "asi": "g.s",
    "si": "cm",
    "dsi": "cm.s",
    "masi": "g.s",
    "ds575": "s",
    "ds595": "s",
    "bracketed_duration": "s",
    "uniform_duration": "s",
    "arms": "g",
    "vrms": "cm/s",
    "drms": "cm",
    "ic": "g**1.5 s**0.5",
    "tm": "s",
    "tp_predominant": "s",
}


def to_g(acc: np.ndarray, unit: str = "g", dtype=np.float32) -> np.ndarray:
    """Convert an acceleration trace to g.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration values.
    unit : str, optional
        Unit of ``acc``, one of :data:`ACCELERATION_UNITS`. Default ``"g"``,
        which is what a PEER .AT2 file holds.
    dtype : numpy.dtype, optional
        Working float type, float32 by default. See
        :data:`djura.signal_processing.ims.PRECISIONS`.

    Returns
    -------
    numpy.ndarray
        Acceleration [g], in ``dtype``.

    Raises
    ------
    ValueError
        If ``unit`` is not recognised.
    """
    dtype = np.dtype(dtype)
    key = str(unit).strip().lower()
    if key not in ACCELERATION_UNITS:
        raise ValueError(
            "unknown acceleration unit %r; expected one of %s"
            % (unit, sorted(set(ACCELERATION_UNITS))))
    factor = ACCELERATION_UNITS[key]
    values = np.asarray(acc, dtype=dtype).ravel()
    if factor == 1.0:
        return values
    return values * dtype.type(factor)
