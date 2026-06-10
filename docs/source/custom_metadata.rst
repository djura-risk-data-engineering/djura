Using a custom metadata file
============================

The record-selection routines in :mod:`djura.record_selection` never read a
ground motion database directly. They operate on a single, in-memory
**metadata object**: a plain Python ``dict`` of NumPy arrays in which the
record-level fields and intensity-measure (IM) values are pre-populated in a
common schema. The bundled ``NGA_W2_v2.pickle`` is just one realisation of
that schema.

This means a *new* database can be incorporated by mapping its records onto
the same schema. Once the metadata fields and IM values are populated in the
common format, the selection routines run without modification. The remainder
of this page documents that schema and shows how to build and use your own
metadata file.

.. note::

   You do **not** need to redistribute or modify the package. A custom
   metadata file is supplied at run time (see `Pointing djura at your file`_),
   so the bundled dataset is simply replaced by yours.

The schema at a glance
----------------------

The metadata object is a flat dictionary. Conceptually its keys fall into four
groups:

#. **Record-level scalar fields** — one value per record, stored as a 1-D
   array of length ``N`` (the number of records). These are identifiers, source
   and site causal parameters, waveform metadata, and period-independent IMs.
#. **Period vectors** — 1-D arrays giving the periods at which the
   period-dependent IMs are tabulated.
#. **Period-dependent IM arrays** — 2-D arrays of shape
   ``(N, len(period_vector))``, one row per record.
#. **Global scalars** — e.g. the damping ratio of the response spectra.

Every record-level array must share the same length ``N`` and the same row
ordering, so that row ``i`` refers to the same record across all keys.

Required record-level fields
----------------------------

The following keys are read by the selection and reporting routines and must
be present. Each is a 1-D array of length ``N``.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Key
     - dtype
     - Meaning
   * - ``RSN``
     - int
     - Unique record identifier (the record sequence number). Used as the
       primary key throughout selection.
   * - ``EQID``
     - int
     - Event identifier. Records sharing an ``EQID`` come from the same
       earthquake (used to limit how many records are drawn from one event).
   * - ``Filename_1``
     - str
     - Filename of the first horizontal component.
   * - ``Filename_2``
     - str
     - Filename of the second horizontal component.
   * - ``Filename_vert``
     - str / object
     - Filename of the vertical component (may be empty strings if unused).
   * - ``EQ_name``
     - str
     - Earthquake name.
   * - ``EQ_year``
     - int
     - Year of the earthquake.
   * - ``Station_name``
     - str
     - Recording station name.
   * - ``magnitude``
     - float
     - Moment magnitude.
   * - ``mechanism``
     - int
     - Fault mechanism code (see `Fault mechanism encoding`_).
   * - ``Rjb``
     - float
     - Joyner–Boore distance [km].
   * - ``Rrup``
     - float
     - Rupture distance [km].
   * - ``Vs30``
     - float
     - Time-averaged shear-wave velocity to 30 m [m/s].
   * - ``lowest_usable_freq``
     - float
     - Lowest usable frequency [Hz]; used to screen records against the
       required period range.
   * - ``dt``
     - float
     - Time step of the waveform [s].
   * - ``duration``
     - float
     - Record duration [s].
   * - ``npts``
     - int
     - Number of samples in the waveform.

Optional causal-context fields
------------------------------

Additional fields are read only if you pass limits on them via
``context_limits`` during selection; otherwise they are ignored. Provide them
when you want to filter on them. Common ones include ``Z1``, ``Z1pt5``,
``Z2pt5``, ``D_hyp``, ``Ds575``, ``Ds595``, ``Tp``, ``rake``, ``dip``,
``strike``, ``Ztor``, ``Rx``, ``rup_width`` and ``soil_NEHRP``. See
:data:`djura.record_selection.constants.DB_CAUSAL_PARS` for the full list the
package recognises by name.

Fault mechanism encoding
------------------------

``mechanism`` is an integer code mapped by
:data:`djura.record_selection.constants.MECHANISM_MAP`:

.. list-table::
   :header-rows: 1
   :widths: 10 40

   * - Code
     - Mechanism
   * - ``0``
     - strike-slip fault
   * - ``1``
     - normal fault
   * - ``2``
     - reverse fault
   * - ``3``
     - reverse/oblique fault
   * - ``4``
     - normal/oblique fault

Intensity-measure fields
------------------------

IM values follow a strict naming convention so the selection code can locate
them automatically. For an IM named ``<IM>``:

**Period-independent IMs** (``PGA``, ``PGV``, ``IA``, ``Ds575``, ``Ds595``)
are stored directly under the IM name as a 1-D array of length ``N``::

   metadata["PGA"]      # shape (N,)
   metadata["Ds595"]    # shape (N,)

**Period-dependent IMs** (``SA``, ``Sa_avg2``, ``Sa_avg3``, ``FIV3``) require
a period vector plus per-component 2-D arrays of shape ``(N, len(periods))``:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Key pattern
     - Shape
     - Meaning
   * - ``<IM>_1``
     - ``(N, P)``
     - First horizontal component
   * - ``<IM>_2``
     - ``(N, P)``
     - Second horizontal component
   * - ``<IM>_RotD50``
     - ``(N, P)``
     - RotD50 component (optional, see below)
   * - ``<IM>_RotD100``
     - ``(N, P)``
     - RotD100 component (optional, see below)

Here ``P`` is the length of the corresponding period vector. The period
vectors expected by the bundled IMs are:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - IM family
     - Period vector key
     - Used by
   * - ``SA``
     - ``Periods_SA``
     - ``SA_1``, ``SA_2``, ``SA_RotD50``, ``SA_RotD100``, ``SA_vert``
   * - ``Sa_avg2`` / ``Sa_avg3``
     - ``Periods_Sa_avg``
     - ``Sa_avg2_1`` … ``Sa_avg3_RotD100``
   * - ``FIV3``
     - ``Periods_FIV3``
     - ``FIV3_1``, ``FIV3_2``

.. important::

   Periods are matched by value, and the package rounds them to 5 decimal
   places. Intermediate periods requested during selection are obtained by
   **linear interpolation** along the period axis, so the period vectors should
   span the range you intend to use. Each row of an IM array corresponds to the
   record in the same row of ``Filename_1`` / ``RSN``.

Component definitions
~~~~~~~~~~~~~~~~~~~~~~

When two horizontal components are available, the selector can build the
following component definitions:

- ``geomean`` — :math:`\sqrt{IM_1 \cdot IM_2}`
- ``srss`` — :math:`\sqrt{IM_1^2 + IM_2^2}`
- ``arithmeticmean`` — :math:`(IM_1 + IM_2)/2`
- ``rotd50`` — read directly from ``<IM>_RotD50``
- ``rotd100`` — read directly from ``<IM>_RotD100``

``geomean``, ``srss`` and ``arithmeticmean`` are computed on the fly from
``<IM>_1`` and ``<IM>_2``, so they require no extra storage. ``rotd50`` and
``rotd100`` are **read directly** from precomputed arrays — supply
``<IM>_RotD50`` / ``<IM>_RotD100`` only if you intend to select on those
components.

Global scalars
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``damping``
     - Damping ratio of the tabulated response spectra (e.g. ``0.05`` for 5 %).

Minimal working example
------------------------

The snippet below assembles a tiny, schema-conformant metadata file from your
own data. Replace the placeholder arrays with values mapped from your
database. Only ``SA`` is populated here; add other IM families the same way.

.. code-block:: python

   import pickle
   import numpy as np

   N = 100                       # number of records
   periods_sa = np.array(        # period vector for SA
       [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
       dtype="float32")
   P = len(periods_sa)

   metadata = {
       # --- required record-level fields (length N) ---
       "RSN":                np.arange(1, N + 1, dtype="int32"),
       "EQID":               np.zeros(N, dtype="int32"),
       "Filename_1":         np.array([f"REC{i}_H1.txt" for i in range(N)]),
       "Filename_2":         np.array([f"REC{i}_H2.txt" for i in range(N)]),
       "Filename_vert":      np.array([""] * N, dtype=object),
       "EQ_name":            np.array(["MyEvent"] * N, dtype=object),
       "EQ_year":            np.full(N, 2020, dtype="int16"),
       "Station_name":       np.array([f"STA{i}" for i in range(N)]),
       "magnitude":          np.random.uniform(5.0, 7.5, N).astype("float32"),
       "mechanism":          np.zeros(N, dtype="int16"),       # 0 = strike-slip
       "Rjb":                np.random.uniform(1, 100, N).astype("float32"),
       "Rrup":               np.random.uniform(1, 100, N).astype("float32"),
       "Vs30":               np.random.uniform(180, 760, N).astype("float32"),
       "lowest_usable_freq": np.full(N, 0.1, dtype="float32"),
       "dt":                 np.full(N, 0.005, dtype="float32"),
       "duration":           np.full(N, 30.0, dtype="float32"),
       "npts":               np.full(N, 6000, dtype="int32"),

       # --- period vector + SA component arrays (shape N x P) ---
       "Periods_SA":         periods_sa,
       "SA_1":               np.random.uniform(0.01, 1.0, (N, P)).astype("float32"),
       "SA_2":               np.random.uniform(0.01, 1.0, (N, P)).astype("float32"),

       # --- global scalar ---
       "damping":            0.05,
   }

   with open("my_metadata.pickle", "wb") as f:
       pickle.dump(metadata, f)

.. tip::

   Keep the row ordering identical across every array — row ``i`` must refer to
   the same physical record in ``RSN``, ``Filename_1``, ``magnitude``,
   ``SA_1`` and all other keys. This is the single most common source of error
   when mapping a new database.

Validating your file
---------------------

A quick sanity check before using the file in a selection run:

.. code-block:: python

   import pickle
   import numpy as np

   with open("my_metadata.pickle", "rb") as f:
       m = pickle.load(f)

   N = len(m["RSN"])
   for key, val in m.items():
       if isinstance(val, np.ndarray) and val.ndim == 1 and key not in (
               "Periods_SA", "Periods_Sa_avg", "Periods_FIV3"):
           assert len(val) == N, f"{key} has length {len(val)}, expected {N}"

   assert m["SA_1"].shape == (N, len(m["Periods_SA"]))
   print(f"OK: {N} records, {len(m['Periods_SA'])} SA periods")

Pointing djura at your file
---------------------------

Custom metadata is supplied at run time through the ``DJURA_METADATA_PATH``
environment variable. When set, it fully bypasses the bundled download and the
selection routines use your file instead:

.. code-block:: bash

   # bash / macOS / Linux
   export DJURA_METADATA_PATH=/path/to/my_metadata.pickle

   # PowerShell
   $env:DJURA_METADATA_PATH = "C:\path\to\my_metadata.pickle"

   # CMD
   set DJURA_METADATA_PATH=C:\path\to\my_metadata.pickle

After that, the API is unchanged:

.. code-block:: python

   from djura.record_selection import GCIM

   gcim = GCIM()            # loads my_metadata.pickle via DJURA_METADATA_PATH
   gcim.get_metadata_parameters()   # inspect the keys it loaded

Both ``.pickle``/``.pkl`` and ``.npz`` files are recognised by the lower-level
reader; the pickled ``dict`` form shown above is the simplest and is what the
bundled dataset uses.

.. note::

   Set the environment variable **before** importing ``djura`` (or before the
   first call that loads the dataset), since the metadata is loaded at most
   once per process and cached.

What you can omit
-----------------

You only need to populate the fields and IMs your selection actually uses:

- IM families you never select on can be left out entirely (e.g. skip all
  ``FIV3_*`` keys and ``Periods_FIV3`` if you do not use ``FIV3``).
- ``RotD50`` / ``RotD100`` arrays are needed only for those component
  definitions; ``geomean`` works from ``_1`` / ``_2`` alone.
- Optional causal fields are read only when you pass matching
  ``context_limits``.

Conversely, the required record-level fields in the table above should always
be present, because they are used for identification, event grouping, usable
period screening, and the selection report.
