Signal processing
=================

This example computes intensity measures of a recorded ground motion with
``djura.signal_processing``: the peaks, energies and durations of a single
component, the RotDxx measures of a horizontal pair, record conditioning,
and pulse classification.

Requires ``pip install "djura[signal_processing]"``.

Input data
----------

The two horizontal components of NGA-West2 RSN 179 (Imperial Valley 1979,
El Centro Array #4), in PEER ``.AT2`` format:
`tests/sp/assets/ <https://github.com/djura-risk-data-engineering/djura/tree/main/tests/sp/assets>`_

Reading a record
----------------

:func:`~djura.signal_processing.records.read_nga` returns the time step,
the number of points, the header description, the time vector and the
acceleration trace. ESM records are read with
:func:`~djura.signal_processing.records.read_esm`.

.. code-block:: python

   import numpy as np
   from djura import signal_processing as sig

   dt, npts, desc, t, acc1 = sig.read_nga("RSN179_IMPVALL.H_H-E04140.AT2")
   _, _, _, _, acc2 = sig.read_nga("RSN179_IMPVALL.H_H-E04230.AT2")

One component
-------------

A :class:`~djura.signal_processing.gm.Component` holds one acceleration
trace. Scalar measures are properties; spectral measures are methods
taking the periods.

.. code-block:: python

   gm = sig.Component(acc1, dt, unit="g")

   gm.pga, gm.pgv, gm.pgd     # g, cm/s, cm
   gm.ia, gm.cav              # m/s, g.s
   gm.ds575, gm.ds595         # s

   periods = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
   gm.psa(periods)            # pseudo-spectral acceleration, g
   gm.sa_avg([1.0])           # average spectral acceleration, g
   gm.fiv3([1.0])             # filtered incremental velocity, cm/s

Units of every measure are listed in
:data:`djura.signal_processing.units.IM_UNITS`.

A horizontal pair
-----------------

A :class:`~djura.signal_processing.gm.GroundMotion` rotates the two
horizontal components together. :meth:`~djura.signal_processing.gm.GroundMotion.rotd`
takes any measure by name. When several spectral measures are needed,
:meth:`~djura.signal_processing.gm.GroundMotion.precompute` solves the
rotated SDOF problem once and the rest is served from its cache.

.. code-block:: python

   pair = sig.GroundMotion(acc1, acc2, dt, unit="g")
   pair.precompute(periods, sa_avg_periods=[1.0])

   sa50, sa100 = pair.rotd("psa", [50, 100], periods)
   pga50 = pair.rotd("pga", 50)
   ia50 = pair.rotd("ia", 50)

   pair.at_angle("psa", 0.0, periods)   # as-recorded first component
   pair.geomean("psa", periods)         # geometric mean of the two

The SDOF solver defaults to the exact piecewise-linear solution;
``solver="newmark"`` uses Newmark integration instead. ``precision=64``
switches every computation to double precision.

Conditioning
------------

Baseline correction and Butterworth filtering return a new object with the
same settings.

.. code-block:: python

   corrected = pair.baseline_corrected("Linear")
   filtered = corrected.filtered(cut_off=(0.1, 25), filter_order=4)

The same operations are available on plain arrays through
:func:`~djura.signal_processing.processing.baseline_correction` and
:func:`~djura.signal_processing.processing.butterworth_filter`.

Pulse classification
--------------------

The pulse classifier of Shahi and Baker (2014) runs over both horizontal
components on first access.

.. code-block:: python

   pair.tpulse               # pulse period, s; -999 if no pulse is found
   pair.pulse.make_plot()    # needs the plot extra
