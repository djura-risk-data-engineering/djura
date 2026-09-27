# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Ground-motion record processing and intensity measures.

Author: Volkan Ozsarac, University School for Advanced Studies IUSS Pavia.

Read a record and ask it for what you need::

    from djura import signal_processing as sig

    dt, npts, desc, t, acc = sig.read_nga("RSN179_IMPVALL.H_H-E04140.AT2")
    gm = sig.Component(acc, dt, unit="g")
    gm.pga, gm.pgv, gm.ia, gm.ds595       # g, cm/s, m/s, s
    gm.psa([0.2, 1.0, 2.0])               # g

Every measure is its own accessor: spectral ones are methods taking the
periods, scalars are properties. For a pair, name the periods up front so the
rotated SDOF problem is solved once::

    gm = sig.GroundMotion(acc1, acc2, dt, acc_vertical)
    gm.precompute(periods_sa, sa_avg_periods)

    sa50, sa100 = gm.rotd("psa", [50, 100], periods_sa)
    first = gm.at_angle("psa", 0.0, periods_sa)
    pga = gm.rotd("pga", [50, 100])

:meth:`~gm.GroundMotion.rotd` takes any measure by name, and any
callable taking ``(acc, dt)``::

    gm.rotd("ia", 50)
    gm.rotd("ei_rel", 50, periods_sa)
    gm.rotd(sig.ims.cumulative_absolute_velocity)

Measures are returned in the conventional units of
:data:`djura.signal_processing.units.IM_UNITS`.

Modules
-------
``gm``
    :class:`Component`, one acceleration component, and
    :class:`GroundMotion`, the horizontal pair and optional vertical component.
``ims``
    Every intensity measure, stateless: peaks, energies, durations, spectrum
    intensities, FIV3, and the SDOF layer behind the response spectra - the
    response maxima, the rotated response and the Sa_avg grids.
``rotation``
    RotDxx, generalised to any measure.
``processing``
    Baseline correction and Butterworth filtering.
``sdof``, ``pulse``, ``records``
    Newmark and exact solvers, pulse classification, record readers.
"""
from .._extras import require_extra

with require_extra("signal_processing"):
    from . import ims, processing, rotation, units
    from .gm import Component, GroundMotion
    from .ims import DEFAULT_ANGLES
    from .processing import baseline_correction, butterworth_filter
    from .records import read_nga
    from .sdof import lrha, nigam_jennings
    from .units import GRAVITY, G_TO_CM, IM_UNITS

__all__ = [
    "Component", "GroundMotion",
    "baseline_correction", "butterworth_filter", "read_nga", "lrha",
    "nigam_jennings", "GRAVITY", "G_TO_CM", "IM_UNITS",
    "DEFAULT_ANGLES",
    "ims", "processing", "rotation", "units", "cite",
]

__citation__ = (
    "@inproceedings{shahnazaryan2025djuraSP,\n"
    "  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and "
    "O'Reilly, Gerard J.},\n"
    "  title     = {{DJURA Ground Motion Record Selector: "
    "A Software Solution for Earthquake Engineering}},\n"
    "  booktitle = {COMPDYN 2025 Proceedings},\n"
    "  year      = {2025}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the signal_processing submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
