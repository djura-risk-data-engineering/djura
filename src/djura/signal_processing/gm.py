# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""The :class:`Component` record and the :class:`GroundMotion` set of two
horizontals and an optional vertical.

Every intensity measure is its own accessor: spectral ones are methods taking
the periods, scalars are properties::

    c.psa(periods)      c.pga
    c.sv_rel(periods)   c.ia

Peak responses are cached per period, not the response histories. They are
computed in two groups: ``sd_rel``, ``psv`` and ``psa``, and the other seven
spectral measures. Asking for any member computes its whole group.

:meth:`GroundMotion.rotd` computes RotDxx of any measure.
"""
from functools import cached_property
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

from . import ims, rotation
from .processing import baseline_correction, butterworth_filter
from .units import GRAVITY, G_TO_CM, to_g

#: Reductions the cheap group provides. ``psv`` and ``psa`` follow from ``sd``.
CORE_FIELDS = ("sd_rel",)

#: Reductions the full group provides, on top of ``sd``.
EXTRA_FIELDS = ("sv_rel", "sa_rel", "sa_abs", "sv_abs", "sd_abs", "ei_rel",
                "ei_abs")

#: Peak values of the trace, which rotate without an oscillator.
PEAK_FIELDS = ("pga", "pgv", "pgd")

#: Measures derived from a reduction. They are rotated by transforming the
#: reduction at every angle before taking the percentile.
DERIVED_FIELDS = ("v_eq",)

#: Short names accepted wherever a measure is named, following NGA-West2.
ALIASES = {"sd": "sd_rel", "sv": "psv", "sa": "psa"}

#: Spectral measures with a rotated route, so they can be read off one angle.
ROTATED_FIELDS = (CORE_FIELDS + EXTRA_FIELDS + DERIVED_FIELDS
                  + ("psv", "psa") + tuple(ALIASES))

#: Measures with their own rotated routine, dispatched by
#: :meth:`GroundMotion.rotd` to the private method of the same name.
WINDOWED_FIELDS = ("sa_avg", "fiv3")

#: Spectrum intensities, as ``name -> (rotated field, integrator)``, each
#: integrated from the rotated spectrum at every angle.
INTENSITY_FIELDS = {
    "si": ("psv", ims.housner_intensity),
    "asi": ("psa", ims.acceleration_spectrum_intensity),
    "masi": ("psa", ims.modified_acceleration_spectrum_intensity),
    "dsi": ("sd_rel", ims.displacement_spectrum_intensity),
}


def _to_package_units(maxima: dict) -> dict:
    """Convert a reduction dict from g-based units to the package's."""
    cm, energy = G_TO_CM, GRAVITY ** 2
    scale = {"sd_rel": cm, "sv_rel": cm, "sv_abs": cm,
             "sd_abs": cm, "ei_rel": energy, "ei_abs": energy}
    return {name: values * values.dtype.type(scale.get(name, 1.0))
            for name, values in maxima.items()}


def _lookup(cache: dict, periods: np.ndarray, full: bool,
            dtype=np.float32):
    """A cached reduction covering ``periods``, sliced; None if absent."""
    for (known_bytes, known_full), entry in cache.items():
        if full and not known_full:
            continue
        known = np.frombuffer(known_bytes, dtype=dtype)
        position = np.clip(np.searchsorted(known, periods), 0, known.size - 1)
        if np.all(known[position] == periods):
            return {name: values[position] for name, values in entry.items()}
    return None


def _restore(stored: dict, order: np.ndarray) -> dict:
    """Undo the sort applied before solving."""
    restored = {}
    for name, values in stored.items():
        back = np.empty_like(values)
        back[order] = values
        restored[name] = back
    return restored


class Component:
    """A single acceleration component and the measures computed from it.

    Parameters
    ----------
    acc : numpy.ndarray
        Acceleration trace.
    dt : float
        Time step [s].
    unit : str, optional
        Unit of ``acc``, ``"g"`` by default. See
        :data:`djura.signal_processing.units.ACCELERATION_UNITS`.
    damping : float, optional
        Damping ratio for every spectral quantity, 0.05 by default.
    name : str, optional
        Label carried for reporting.
    solver : {'newmark', 'exact'}, optional
        SDOF solver, :data:`djura.signal_processing.ims.DEFAULT_SOLVER` if
        omitted.
    min_points : int, optional
        Samples per oscillator period; the record is interpolated where it is
        coarser. :data:`djura.signal_processing.ims.MIN_POINTS_PER_PERIOD` if
        omitted, ``None`` to solve at the record's own step.
    precision : int, optional
        Float precision, 32 (default) or 64; see
        :data:`djura.signal_processing.ims.PRECISIONS`.

    Examples
    --------
    >>> c = Component(acc, dt, unit="g")
    >>> c.pga, c.ia, c.ds595
    >>> c.psa([0.2, 1.0, 2.0])
    """

    def __init__(self, acc: np.ndarray, dt: float, unit: str = "g",
                 damping: float = 0.05, name: Optional[str] = None,
                 solver: Optional[Literal["newmark", "exact"]] = None,
                 min_points=ims.UNSET, precision: int = 32):
        self.precision = precision
        self.dtype = ims._dtype(precision)
        self.acc = to_g(acc, unit, self.dtype)
        self.dt = float(dt)
        self.damping = float(damping)
        self.name = name
        self.solver = ims.DEFAULT_SOLVER if solver is None else solver
        self.min_points = min_points
        if self.solver not in ims.SOLVERS:
            raise ValueError("solver must be one of %s; got %r"
                             % (ims.SOLVERS, self.solver))
        if self.acc.size < 2:
            raise ValueError("a trace needs at least two samples")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self._cache = {}

    def __repr__(self) -> str:
        label = "" if self.name is None else " %r" % self.name
        return ("<Component%s npts=%d dt=%g duration=%.2f s PGA=%.4g g "
                "solver=%s>" % (label, self.npts, self.dt, self.duration,
                                self.pga, self.solver))

    @property
    def npts(self) -> int:
        """Number of samples."""
        return int(self.acc.size)

    @property
    def duration(self) -> float:
        """Record length [s], counted as ``npts * dt``."""
        return self.npts * self.dt

    @cached_property
    def time(self) -> np.ndarray:
        """Time vector [s]."""
        return (self.dt * np.arange(self.npts)).astype(self.dtype)

    @cached_property
    def vel(self) -> np.ndarray:
        """Ground velocity history [cm/s]."""
        return ims.velocity(self.acc, self.dt)

    @cached_property
    def disp(self) -> np.ndarray:
        """Ground displacement history [cm]."""
        return ims.displacement(self.vel, self.dt)

    @cached_property
    def arias(self) -> np.ndarray:
        """Cumulative Arias intensity against time [m/s]."""
        return ims.arias_series(self.acc, self.dt)

    @cached_property
    def pga(self) -> float:
        """Peak ground acceleration [g]."""
        return ims.peak_ground_acceleration(self.acc)

    @cached_property
    def pgv(self) -> float:
        """Peak ground velocity [cm/s]."""
        return ims.peak_ground_velocity(self.vel)

    @cached_property
    def pgd(self) -> float:
        """Peak ground displacement [cm]."""
        return ims.peak_ground_displacement(self.disp)

    @cached_property
    def ia(self) -> float:
        """Arias intensity [m/s]."""
        return float(self.arias[-1])

    @cached_property
    def cav(self) -> float:
        """Cumulative absolute velocity [g.s]."""
        return ims.cumulative_absolute_velocity(self.acc, self.dt)

    @cached_property
    def arms(self) -> float:
        """Root mean square acceleration [g]."""
        return ims.root_mean_square(self.acc, self.dt)

    @cached_property
    def vrms(self) -> float:
        """Root mean square velocity [cm/s]."""
        return ims.root_mean_square(self.vel, self.dt)

    @cached_property
    def drms(self) -> float:
        """Root mean square displacement [cm]."""
        return ims.root_mean_square(self.disp, self.dt)

    @cached_property
    def ic(self) -> float:
        """Characteristic intensity over the full record length."""
        return ims.characteristic_intensity(self.arms, self.time[-1])

    def significant_duration(self, lower: float = 0.05,
                             upper: float = 0.95
                             ) -> Tuple[float, float, float]:
        """Significant duration between two fractions of the Arias intensity.

        Parameters
        ----------
        lower, upper : float, optional
            Fractions of the total, 0.05 and 0.95 by default.

        Returns
        -------
        start, end, duration : float
            All in seconds.
        """
        return ims.significant_duration(self.arias, self.dt, lower, upper)

    @cached_property
    def ds575(self) -> float:
        """Significant duration, 5 to 75 per cent of Arias intensity [s]."""
        return self.significant_duration(0.05, 0.75)[2]

    @cached_property
    def ds595(self) -> float:
        """Significant duration, 5 to 95 per cent of Arias intensity [s]."""
        return self.significant_duration(0.05, 0.95)[2]

    def bracketed_duration(self, threshold: float = 0.05) -> float:
        """Time between first and last excursion beyond a threshold [s]."""
        return ims.bracketed_duration(self.acc, self.dt, threshold)

    def bracketed_interval(self, threshold: float = 0.05
                           ) -> Tuple[float, float]:
        """First and last instants beyond a threshold [s]."""
        return ims.bracketed_interval(self.acc, self.dt, threshold)

    def uniform_duration(self, threshold: float = 0.05) -> float:
        """Total time spent beyond a threshold [s]."""
        return ims.uniform_duration(self.acc, self.dt, threshold)

    @cached_property
    def fas(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fourier amplitude spectrum, ``(frequency [Hz], amplitude
        [g.s])``."""
        return ims.fourier_amplitude_spectrum(self.acc, self.dt, self.dtype)

    @cached_property
    def pas(self) -> Tuple[np.ndarray, np.ndarray]:
        """Power amplitude spectrum, ``(frequency [Hz], power)``."""
        return ims.power_amplitude_spectrum(self.acc, self.dt, self.dtype)

    @cached_property
    def tm(self) -> float:
        """Mean period [s]."""
        return ims.mean_period(*self.fas)

    def _reductions(self, periods: np.ndarray, full: bool) -> dict:
        """Cached peak responses for ``periods``, in the package's units."""
        periods = ims._as_periods(periods, self.dtype)
        order = np.argsort(periods)
        ordered = periods[order]

        stored = _lookup(self._cache, ordered, full, self.dtype)
        if stored is None:
            common = dict(damping=self.damping, solver=self.solver,
                          min_points=self.min_points,
                          precision=self.precision)
            if full:
                stored = ims.response_maxima(self.acc, self.dt, ordered,
                                             **common)
            else:
                stored = {"sd_rel": ims.spectral_displacement(
                    self.acc, self.dt, ordered, **common)}
            stored = _to_package_units(stored)
            self._cache[(ordered.tobytes(), full)] = stored
        return _restore(stored, order)

    def _field(self, periods: np.ndarray, name: str) -> np.ndarray:
        """One reduction by name, computing its group if needed."""
        return self._reductions(periods, name in EXTRA_FIELDS)[name]

    def sd_rel(self, periods: np.ndarray) -> np.ndarray:
        """Spectral (relative) displacement [cm]."""
        return self._field(periods, "sd_rel")

    def sd(self, periods: np.ndarray) -> np.ndarray:
        """Spectral displacement [cm]; alias of :meth:`sd_rel`."""
        return self.sd_rel(periods)

    def psv(self, periods: np.ndarray) -> np.ndarray:
        """Pseudo-spectral velocity [cm/s]."""
        periods = ims._as_periods(periods, self.dtype)
        omega = (2.0 * np.pi / periods).astype(self.dtype)
        return self.sd_rel(periods) * omega

    def sv(self, periods: np.ndarray) -> np.ndarray:
        """Spectral velocity [cm/s]; alias of :meth:`psv`."""
        return self.psv(periods)

    def psa(self, periods: np.ndarray) -> np.ndarray:
        """Pseudo-spectral acceleration [g]."""
        periods = ims._as_periods(periods, self.dtype)
        omega = (2.0 * np.pi / periods).astype(self.dtype)
        return self.psv(periods) * omega / self.dtype.type(G_TO_CM)

    def sa(self, periods: np.ndarray) -> np.ndarray:
        """Spectral acceleration [g]; alias of :meth:`psa`."""
        return self.psa(periods)

    def sv_rel(self, periods: np.ndarray) -> np.ndarray:
        """Spectral (relative) velocity [cm/s]."""
        return self._field(periods, "sv_rel")

    def sa_rel(self, periods: np.ndarray) -> np.ndarray:
        """Relative acceleration [g]."""
        return self._field(periods, "sa_rel")

    def sa_abs(self, periods: np.ndarray) -> np.ndarray:
        """Absolute acceleration [g]."""
        return self._field(periods, "sa_abs")

    def sv_abs(self, periods: np.ndarray) -> np.ndarray:
        """Absolute velocity [cm/s]."""
        return self._field(periods, "sv_abs")

    def sd_abs(self, periods: np.ndarray) -> np.ndarray:
        """Absolute displacement [cm]."""
        return self._field(periods, "sd_abs")

    def ei_rel(self, periods: np.ndarray) -> np.ndarray:
        """Relative input energy per unit mass [m2/s2].

        ``-integral(a_g du)``, from the relative energy equation of Uang and
        Bertero (1990).

        References
        ----------
        Uang, C. M., and Bertero, V. V. (1990). Evaluation of seismic energy
        in structures. Earthquake Engineering & Structural Dynamics, 19(1),
        77-90.

        Parameters
        ----------
        periods : numpy.ndarray
            Oscillator periods [s].

        Returns
        -------
        numpy.ndarray
            Ei_rel [m2/s2].
        """
        return self._field(periods, "ei_rel")

    def ei_abs(self, periods: np.ndarray) -> np.ndarray:
        """Absolute input energy per unit mass [m2/s2].

        ``integral(a_abs du_g)``, from the absolute energy equation of Uang
        and Bertero (1990).

        References
        ----------
        Uang, C. M., and Bertero, V. V. (1990). Evaluation of seismic energy
        in structures. Earthquake Engineering & Structural Dynamics, 19(1),
        77-90.

        Parameters
        ----------
        periods : numpy.ndarray
            Oscillator periods [s].

        Returns
        -------
        numpy.ndarray
            Ei_abs [m2/s2].
        """
        return self._field(periods, "ei_abs")

    def v_eq(self, periods: np.ndarray) -> np.ndarray:
        """Absolute input energy as an equivalent velocity [cm/s].

        ``V_eq = sqrt(2 Ei_abs)``.

        Parameters
        ----------
        periods : numpy.ndarray
            Oscillator periods [s].

        Returns
        -------
        numpy.ndarray
            V_eq [cm/s].
        """
        return 100.0 * np.sqrt(2.0 * np.maximum(self.ei_abs(periods), 0.0))

    def sa_avg(self, periods: np.ndarray, t_low: float = 0.2,
               t_high: float = 2.0, n_per: int = 10) -> np.ndarray:
        """Average spectral acceleration over ``[t_low T, t_high T]`` [g].

        Parameters
        ----------
        periods : numpy.ndarray
            Anchor periods [s].
        t_low, t_high : float, optional
            Averaging bounds as multiples of the anchor.
        n_per : int, optional
            Periods per window, 10 by default.

        Returns
        -------
        numpy.ndarray
            Sa_avg [g].
        """
        grid, indices = ims.sa_avg_windows(periods, t_low, t_high, n_per,
                                           self.dtype)
        return ims.average_over_windows(self.psa(grid), indices, self.dtype)

    def fiv3(self, periods: np.ndarray, alpha: float = 0.7,
             beta: float = None,
             filtering: Literal["causal", "acausal"] = "causal"
             ) -> np.ndarray:
        """Filtered incremental velocity [cm/s].

        Parameters
        ----------
        periods : numpy.ndarray
            Periods [s].
        alpha, beta, filtering
            See
            :func:`djura.signal_processing.ims.filtered_incremental_velocity`.

        Returns
        -------
        numpy.ndarray
            FIV3 [cm/s].
        """
        return ims.filtered_incremental_velocity(self.acc, self.dt, periods,
                                                 alpha, beta, filtering,
                                                 self.dtype)

    def si(self, periods: np.ndarray, lower: float = 0.1,
           upper: float = 2.5) -> float:
        """Housner intensity [cm]. ``periods`` must span ``[lower, upper]``."""
        periods = ims._as_periods(periods, self.dtype)
        return ims.housner_intensity(self.psv(periods), periods, lower, upper)

    def asi(self, periods: np.ndarray, lower: float = 0.1,
            upper: float = 0.5) -> float:
        """Acceleration spectrum intensity [g.s]."""
        periods = ims._as_periods(periods, self.dtype)
        return ims.acceleration_spectrum_intensity(self.psa(periods), periods,
                                                   lower, upper)

    def masi(self, periods: np.ndarray, lower: float = 0.1,
             upper: float = 2.5) -> float:
        """Modified acceleration spectrum intensity [g.s]."""
        periods = ims._as_periods(periods, self.dtype)
        return ims.modified_acceleration_spectrum_intensity(
            self.psa(periods), periods, lower, upper)

    def dsi(self, periods: np.ndarray, lower: float = 2.0,
            upper: float = 5.0) -> float:
        """Displacement spectrum intensity [cm.s]."""
        periods = ims._as_periods(periods, self.dtype)
        return ims.displacement_spectrum_intensity(
            self.sd_rel(periods), periods, lower, upper)

    def tp(self, periods: np.ndarray) -> float:
        """Predominant period at which the response spectrum peaks [s]."""
        periods = ims._as_periods(periods, self.dtype)
        return ims.predominant_period(self.psa(periods), periods)

    def baseline_corrected(
            self, polynomial_type: Literal["Constant", "Linear", "Quadratic",
                                           "Cubic"] = "Linear"
            ) -> "Component":
        """A new record with a polynomial trend removed."""
        return Component(baseline_correction(self.acc, self.dt,
                                             polynomial_type),
                         self.dt, "g", self.damping, self.name, self.solver,
                         self.min_points, self.precision)

    def filtered(self, cut_off: Union[float, Tuple[float, float]] = (0.1, 25),
                 filter_order: int = 4,
                 filter_type: Literal["lowpass", "highpass", "bandpass",
                                      "bandstop"] = "bandpass",
                 filtering: Literal["causal", "acausal"] = "acausal",
                 alpha_window: float = 0.0) -> "Component":
        """A new record, Butterworth filtered.

        Parameters
        ----------
        cut_off, filter_order, filter_type, filtering, alpha_window
            See :func:`djura.signal_processing.processing.butterworth_filter`.

        Returns
        -------
        Component
            The filtered record.
        """
        return Component(
            butterworth_filter(self.acc, self.dt, cut_off, filter_order,
                               filter_type, filtering, alpha_window),
            self.dt, "g", self.damping, self.name, self.solver,
            self.min_points, self.precision)


class GroundMotion:
    """Two horizontal acceleration traces and, optionally, the vertical.

    The components share the same time step, damping, solver and
    ``min_points``, and are available as :attr:`first`, :attr:`second` and
    :attr:`vertical`.

    Parameters
    ----------
    first, second : array_like
        As-recorded horizontal acceleration traces. The shorter is
        zero-padded where the two are rotated together.
    dt : float
        Time step [s].
    vertical : array_like, optional
        Carried for convenience; takes no part in any rotation.
    unit : str, optional
        Unit of the traces, ``"g"`` by default. See
        :data:`djura.signal_processing.units.ACCELERATION_UNITS`.
    damping : float, optional
        Damping ratio for every spectral quantity, 0.05 by default.
    solver : {'newmark', 'exact'}, optional
        SDOF solver, :data:`djura.signal_processing.ims.DEFAULT_SOLVER` if
        omitted.
    min_points : int, optional
        Samples per oscillator period; see :class:`Component`.
    precision : int, optional
        Float precision, 32 (default) or 64; see
        :data:`djura.signal_processing.ims.PRECISIONS`.
    angles : numpy.ndarray, optional
        Rotation angles [deg],
        :data:`djura.signal_processing.ims.DEFAULT_ANGLES` if omitted.

    Examples
    --------
    >>> gm = GroundMotion(acc1, acc2, dt)
    >>> gm = GroundMotion(acc1, acc2, dt, acc_vertical)
    >>> gm.rotd("psa", 50, periods)
    >>> gm.first.pga, gm.second.ia
    """

    def __init__(self, first, second, dt: float, vertical=None,
                 unit: str = "g", damping: float = 0.05,
                 solver: Optional[Literal["newmark", "exact"]] = None,
                 min_points=ims.UNSET, precision: int = 32,
                 angles: Optional[np.ndarray] = None):
        def build(acc, label):
            """One component, or None where the trace was not supplied."""
            if acc is None:
                return None
            return Component(acc, dt, unit, damping, label, solver,
                             min_points, precision)

        if first is None or second is None:
            raise ValueError("both horizontal components are required")
        self.first = build(first, "first")
        self.second = build(second, "second")
        self.vertical = build(vertical, "vertical")
        self.angles = (ims.DEFAULT_ANGLES if angles is None
                       else np.atleast_1d(np.asarray(angles,
                                                     dtype=self.dtype)))
        self._cache = {}
        self._peaks_cache = {}

    def __repr__(self) -> str:
        return ("<GroundMotion npts=%d dt=%g vertical=%s solver=%s "
                "angles=%d>" % (self.first.npts, self.dt,
                                self.vertical is not None, self.solver,
                                self.angles.size))

    @property
    def dt(self) -> float:
        """Time step [s]."""
        return self.first.dt

    @property
    def damping(self) -> float:
        """Damping ratio of every component."""
        return self.first.damping

    @property
    def solver(self) -> str:
        """SDOF solver of every component."""
        return self.first.solver

    @property
    def min_points(self):
        """Samples per oscillator period, for every component."""
        return self.first.min_points

    @property
    def precision(self) -> int:
        """Float precision of every component, 32 or 64."""
        return self.first.precision

    @property
    def dtype(self) -> np.dtype:
        """Working float type of every component."""
        return self.first.dtype

    @property
    def components(self) -> Tuple[Component, ...]:
        """The two horizontals, then the vertical where there is one."""
        if self.vertical is None:
            return (self.first, self.second)
        return (self.first, self.second, self.vertical)

    @cached_property
    def pulse(self):
        """The pulse classifier, run over both horizontal components.

        Imported on first use, since it needs PyWavelets.

        Returns
        -------
        djura.signal_processing.pulse.PulseClassifier
            With :meth:`classify` already called, so ``get_tp`` and
            ``make_plot`` are ready to use.
        """
        from .pulse import PulseClassifier

        classifier = PulseClassifier(self.first.acc, self.second.acc,
                                     self.dt)
        classifier.classify()
        return classifier

    @cached_property
    def tpulse(self) -> float:
        """Pulse period [s]; -999 where no pulse is identified."""
        return self.pulse.get_tp()

    def _reductions(self, periods: np.ndarray, full: bool) -> dict:
        """Cached peak responses at every angle, in the package's units."""
        periods = ims._as_periods(periods, self.dtype)
        order = np.argsort(periods)
        ordered = periods[order]

        stored = _lookup(self._cache, ordered, full, self.dtype)
        if stored is None:
            common = dict(damping=self.damping, angles=self.angles,
                          solver=self.solver, min_points=self.min_points,
                          precision=self.precision)
            if full:
                stored = ims.rotated_response_maxima(
                    self.first.acc, self.second.acc, self.dt, ordered,
                    **common)
            else:
                stored = {"sd_rel": ims.rotated_spectral_displacement(
                    self.first.acc, self.second.acc, self.dt, ordered,
                    **common)}
            stored = _to_package_units(stored)
            self._cache[(ordered.tobytes(), full)] = stored
        return _restore(stored, order)

    def _rotated_field(self, periods: np.ndarray, name: str) -> np.ndarray:
        """One reduction at every angle, shape ``(n_periods, n_angles)``."""
        periods = ims._as_periods(periods, self.dtype)
        name = ALIASES.get(name, name)
        if name == "v_eq":
            energy = self._reductions(periods, True)["ei_abs"]
            return 100.0 * np.sqrt(2.0 * np.maximum(energy, 0.0))
        if name in ("psv", "psa"):
            omega = (2.0 * np.pi / periods).astype(self.dtype)[:, None]
            psv = self._reductions(periods, False)["sd_rel"] * omega
            if name == "psv":
                return psv
            return psv * omega / self.dtype.type(G_TO_CM)
        return self._reductions(periods, name in EXTRA_FIELDS)[name]

    def precompute(self, periods: Optional[np.ndarray] = None,
                   sa_avg_periods: Optional[np.ndarray] = None,
                   sa_avg_ranges: Tuple[Tuple[float, float], ...] = (
                       (0.2, 2.0), (0.2, 3.0)),
                   n_per: int = 10, full: bool = False) -> None:
        """Solve the rotated SDOF problem once over every period needed.

        Everything asked for afterwards is served by slicing the cache.

        Parameters
        ----------
        periods : numpy.ndarray, optional
            Response-spectrum periods [s].
        sa_avg_periods : numpy.ndarray, optional
            Sa_avg anchor periods [s].
        sa_avg_ranges : sequence of tuple, optional
            ``(t_low, t_high)`` ranges that will be asked for.
        n_per : int, optional
            Periods per averaging window, 10 by default.
        full : bool, optional
            Also compute the reductions in :data:`EXTRA_FIELDS`.
        """
        wanted = []
        if periods is not None:
            wanted.append(ims._as_periods(periods, self.dtype))
        if sa_avg_periods is not None:
            for t_low, t_high in sa_avg_ranges:
                grid, _ = ims.sa_avg_windows(sa_avg_periods, t_low,
                                             t_high, n_per, self.dtype)
                wanted.append(grid)
        if not wanted:
            raise ValueError("nothing to precompute")
        self._reductions(np.unique(np.concatenate(wanted)), full)

    def rotd(self, measure: Union[str, Callable[[np.ndarray, float], object]],
             xx: Union[float, Tuple[float, ...]] = (50, 100),
             *args, angles: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        """RotDxx of any measure.

        Every intensity measure has a RotDxx definition, so this takes any of
        them and routes to the cheapest implementation available:

        ==========================  ===========================  ===========
        measure                     route                        cost
        ==========================  ===========================  ===========
        ``sd_rel``, ``psv``, ``psa``convex hull                  1 hull
        the seven extras            convex hull, quadratic form  6 hulls
        ``pga``, ``pgv``, ``pgd``   rotate the integrated series 1 rotation
        ``sa_avg``                  slice the cached grid        no solve
        ``fiv3``                    filter once per period       2 filters
        ``si``, ``asi``, ``dsi``    integrate the rotated        no solve
        anything else               evaluate per angle           180 evals
        ==========================  ===========================  ===========

        ``pga``, ``pgv`` and ``pgd`` share one cached rotation.

        Parameters
        ----------
        measure : str or callable
            An accessor name on :class:`Component`, or a callable invoked as
            ``measure(acc, dt)`` with acceleration in g.
        xx : float or sequence of float, optional
            Percentiles, ``(50, 100)`` by default.
        *args, **kwargs
            Passed to the named accessor, e.g. the periods it needs.
        angles : numpy.ndarray, optional
            Angles [deg]; the pair's own set if omitted.

        Returns
        -------
        numpy.ndarray
            The requested percentiles.

        Examples
        --------
        >>> pair.rotd("psa", 50, periods)
        >>> pair.rotd("ei_rel", 50, periods)
        >>> pair.rotd("ia", 50)
        >>> pair.rotd("si", 50, periods)
        >>> pair.rotd("pgv", 50)
        >>> pair.rotd("sa_avg", 50, periods, t_high=3.0)
        """
        if isinstance(measure, str) and angles is None:
            if measure in ROTATED_FIELDS and len(args) == 1 and not kwargs:
                return rotation.percentiles(
                    self._rotated_field(args[0], measure), xx)
            if measure in PEAK_FIELDS and not args and not kwargs:
                return self._peaks(xx)[PEAK_FIELDS.index(measure)]
            if measure in WINDOWED_FIELDS and args:
                return getattr(self, "_" + measure)(args[0], xx, *args[1:],
                                                    **kwargs)
            if measure in INTENSITY_FIELDS and args:
                field, integrate = INTENSITY_FIELDS[measure]
                periods = ims._as_periods(args[0], self.dtype)
                return rotation.percentiles(
                    integrate(self._rotated_field(periods, field), periods,
                              *args[1:], **kwargs), xx)

        if isinstance(measure, str):
            name, damping, solver = measure, self.damping, self.solver
            min_points, precision = self.min_points, self.precision

            def measure(acc, dt):
                """The named measure on the rotated trace."""
                component = Component(acc, dt, "g", damping, solver=solver,
                                      min_points=min_points,
                                      precision=precision)
                attribute = getattr(component, name)
                return (attribute(*args, **kwargs) if callable(attribute)
                        else attribute)

        return rotation.rotated_scalar(
            self.first.acc, self.second.acc, self.dt, measure, xx,
            self.angles if angles is None else angles)

    def at_angle(self, measure: str, angle: float, periods: np.ndarray
                 ) -> np.ndarray:
        """One spectral measure in a single direction.

        With the default angles, 0 and 90 degrees are the two as-recorded
        components.

        Parameters
        ----------
        measure : str
            One of :data:`ROTATED_FIELDS`.
        angle : float
            Rotation angle [deg]; must be one of :attr:`angles`.
        periods : numpy.ndarray
            Oscillator periods [s].

        Returns
        -------
        numpy.ndarray
            The measure in that direction.

        Examples
        --------
        >>> pair.at_angle("psa", 0.0, periods)     # component 1
        >>> pair.at_angle("psa", 90.0, periods)    # component 2
        """
        if measure not in ROTATED_FIELDS:
            raise ValueError("%r has no rotated route, so it cannot be read "
                             "off a single angle; expected one of %s"
                             % (measure, list(ROTATED_FIELDS)))
        matches = np.flatnonzero(np.isclose(self.angles, angle))
        if matches.size == 0:
            raise ValueError("angle %g degrees is not in the pair's set, "
                             "which runs %g to %g"
                             % (angle, self.angles[0], self.angles[-1]))
        return self._rotated_field(periods, measure)[:, int(matches[0])]

    def _peaks(self, xx: Union[float, Tuple[float, ...]]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cached RotDxx ``(PGA [g], PGV [cm/s], PGD [cm])``."""
        key = np.atleast_1d(np.asarray(xx, dtype=float)).tobytes()
        if key not in self._peaks_cache:
            self._peaks_cache[key] = rotation.rotated_peaks(
                self.first.acc, self.second.acc, self.dt, xx, self.angles)
        return self._peaks_cache[key]

    def _sa_avg(self, periods: np.ndarray,
                xx: Union[float, Tuple[float, ...]] = (50, 100),
                t_low: float = 0.2, t_high: float = 2.0,
                n_per: int = 10) -> np.ndarray:
        """RotDxx average spectral acceleration [g], behind
        ``rotd("sa_avg", ...)``.

        Sa_avg is computed in each direction before taking the percentile,
        not averaged from the RotDxx spectrum.

        Parameters
        ----------
        periods : numpy.ndarray
            Anchor periods [s].
        xx : float or sequence of float, optional
            Percentiles, ``(50, 100)`` by default.
        t_low, t_high, n_per
            See :meth:`Component.sa_avg`.

        Returns
        -------
        numpy.ndarray
            Sa_avg [g], one row per percentile.
        """
        grid, indices = ims.sa_avg_windows(periods, t_low, t_high, n_per,
                                           self.dtype)
        psa = self._rotated_field(grid, "psa")
        return rotation.percentiles(
            ims.average_over_windows(psa, indices, self.dtype), xx)

    def component_sa_avg(self, periods: np.ndarray, t_low: float = 0.2,
                         t_high: float = 2.0,
                         n_per: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Both components' own Sa_avg [g], from the shared rotation.

        Gives the same values as :meth:`Component.sa_avg` on :attr:`first`
        and :attr:`second`, without solving the SDOF problem again.

        Parameters
        ----------
        periods : numpy.ndarray
            Anchor periods [s].
        t_low, t_high, n_per
            See :meth:`Component.sa_avg`.

        Returns
        -------
        first, second : numpy.ndarray
        """
        grid, indices = ims.sa_avg_windows(periods, t_low, t_high, n_per,
                                           self.dtype)
        return tuple(
            ims.average_over_windows(self.at_angle("psa", angle, grid),
                                     indices, self.dtype)
            for angle in (0.0, 90.0))

    def _fiv3(self, periods: np.ndarray,
              xx: Union[float, Tuple[float, ...]] = (50, 100),
              alpha: float = 0.7, beta: Optional[float] = None,
              filtering: Literal["causal", "acausal"] = "causal"
              ) -> np.ndarray:
        """RotDxx filtered incremental velocity [cm/s], behind
        ``rotd("fiv3", ...)``.

        Parameters
        ----------
        periods : numpy.ndarray
            Periods [s].
        xx : float or sequence of float, optional
            Percentiles, ``(50, 100)`` by default.
        alpha, beta, filtering
            See
            :func:`djura.signal_processing.ims.filtered_incremental_velocity`.

        Returns
        -------
        numpy.ndarray
            FIV3 [cm/s], one row per percentile.
        """
        return rotation.percentiles(
            ims.rotated_filtered_incremental_velocity(
                self.first.acc, self.second.acc, self.dt, periods, alpha,
                beta, filtering, self.angles, self.dtype), xx)

    def geomean(self, name: str, *args, **kwargs) -> np.ndarray:
        """Geometric mean of a named measure over the two components.

        Parameters
        ----------
        name : str
            Attribute or method of :class:`Component`.
        *args, **kwargs
            Passed through where ``name`` is a method.

        Returns
        -------
        numpy.ndarray
            Geometric mean, in the measure's own unit.

        Examples
        --------
        >>> pair.geomean("pga")
        >>> pair.geomean("psa", periods)
        """
        values = []
        for component in (self.first, self.second):
            attribute = getattr(component, name)
            values.append(attribute(*args, **kwargs) if callable(attribute)
                          else attribute)
        return rotation.geometric_mean(*values)

    def baseline_corrected(
            self, polynomial_type: Literal["Constant", "Linear", "Quadratic",
                                           "Cubic"] = "Linear"
            ) -> "GroundMotion":
        """A new set with a polynomial trend removed from every component.

        Parameters
        ----------
        polynomial_type : {'Constant', 'Linear', 'Quadratic', 'Cubic'}
            Trend removed; see
            :func:`djura.signal_processing.processing.baseline_correction`.

        Returns
        -------
        GroundMotion
            The corrected set, carrying this one's settings and angles.
        """
        first, second, vertical = (
            None if c is None
            else baseline_correction(c.acc, self.dt, polynomial_type)
            for c in (self.first, self.second, self.vertical))
        return GroundMotion(first, second, self.dt, vertical, unit="g",
                            damping=self.damping, solver=self.solver,
                            min_points=self.min_points,
                            precision=self.precision, angles=self.angles)

    def filtered(self, cut_off: Union[float, Tuple[float, float]] = (0.1, 25),
                 filter_order: int = 4,
                 filter_type: Literal["lowpass", "highpass", "bandpass",
                                      "bandstop"] = "bandpass",
                 filtering: Literal["causal", "acausal"] = "acausal",
                 alpha_window: float = 0.0) -> "GroundMotion":
        """A new set, Butterworth filtered, every component alike.

        Parameters
        ----------
        cut_off, filter_order, filter_type, filtering, alpha_window
            See :func:`djura.signal_processing.processing.butterworth_filter`.

        Returns
        -------
        GroundMotion
            The filtered set, carrying this one's settings and angles.
        """
        first, second, vertical = (
            None if c is None
            else butterworth_filter(c.acc, self.dt, cut_off, filter_order,
                                    filter_type, filtering, alpha_window)
            for c in (self.first, self.second, self.vertical))
        return GroundMotion(first, second, self.dt, vertical, unit="g",
                            damping=self.damping, solver=self.solver,
                            min_points=self.min_points,
                            precision=self.precision, angles=self.angles)
