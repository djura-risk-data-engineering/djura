Exact record selection from an OpenQuake datastore
===================================================

This page documents the *exact* approach to conditional record selection:
every rupture scenario of an
`OpenQuake Engine <https://github.com/gem/oq-engine>`_
probabilistic seismic hazard assessment (PSHA) calculation is
carried into the GCIM target individually, with the full rupture and
distance context the engine used to compute the hazard, and weighted by its
own logic-tree weight and rate of occurrence. No magnitude-distance
binning, no representative scenario, no interpolation of the causal
parameters.

The case lives in
`tests/rs/exact <https://github.com/djura-risk-data-engineering/djura/tree/main/tests/rs/exact>`_
and is driven by one script,
`exact_rs.py <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/exact/exact_rs.py>`_.

.. note::

   **This is not how record selection normally works in djura, and none of
   it is required to select records.** The OQ-engine, the datastore and the
   ``djura-tools`` companion repository are needed *only* for this exact,
   per-rupture-scenario workflow.

   * :doc:`record_selection` selects records from a hazard input written by
     hand, with ``djura`` alone. Nothing else has to be installed.
   * :doc:`record_selection_oq` drives the selection from an OQ
     disaggregation in the *approximate* form: the exported disaggregation
     is read as magnitude-distance bins, and each bin becomes one
     representative scenario.
   * This page is the *exact* form of that same workflow, for when the
     binning is the thing to avoid.

Approximate against exact
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Approximate (:doc:`record_selection_oq`)
     - Exact (this page)
   * - Hazard input
     - Disaggregation exported by the engine, as ``.xml``/``.csv``
     - The calculation datastore, ``calc_<id>.hdf5``
   * - Scenarios
     - One per magnitude-distance(-epsilon) bin, often trimmed to the most
       contributing ones
     - Every rupture the engine considered, per source-model branch
   * - Causal parameters
     - Bin centre for magnitude and distance; the remaining context
       parameters (``rrup``, ``ztor``, ``rx``, ``dip``, ...) are matched
       from a nearby rupture or assumed
     - Read from the engine's own rupture contexts, so exactly the values
       the GMMs were evaluated with
   * - Weights
     - Hazard contribution of the bin
     - Logic-tree leaf weight together with the rupture occurrence rate
       and probabilities
   * - Conditioning level
     - Read off the exported hazard curve
     - Interpolated from the hazard curves stored in the datastore
   * - Needs
     - ``djura`` only
     - ``djura`` + OQ-engine + ``djura-tools``
   * - Web app
     - Yes
     - API only

Files of the example
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path under ``tests/rs/exact``
     - What it is
   * - ``assets/job.ini``
     - The OQ job file. A disaggregation run for a site in Rivisondoli,
       Italy, over 46 intensity measures and 12 probabilities of exceedance
       in 50 years
   * - ``assets/inputs/``
     - The source-model and GMM logic trees (``ssmLT.xml``, ``ssm.xml``,
       ``gmmLT.xml``) and the site models (``rivisondoli.csv``,
       ``laquila.csv``) referenced by ``job.ini``
   * - ``assets/default_data.json``
     - The selection settings only: number of records, seed, KS level,
       maximum scaling factor, component definition. The hazard entries are
       absent on purpose, as they come from the datastore
   * - ``data/calc_9.hdf5``
     - The datastore of that run, copied out of ``oqdata``
   * - ``data/ctx.pickle``
     - The context extracted from the datastore, so the selection can be
       re-run without the OQ-engine installed
   * - ``exact_rs.py``
     - The script: build the context, then select

Step 0: install the OQ-engine
-----------------------------

Needed only to read a datastore, that is, only for ``build_context()``. If
you were given a ``ctx.pickle``, skip to :ref:`exact-selection`.

The engine pins its own versions of numpy, scipy, h5py and shapely, so
install it into a dedicated environment and install ``djura`` into that
same environment afterwards. Consult the
`engine installation guide <https://docs.openquake.org/oq-engine/manual/latest/installation/>`_
for the Python versions a given release supports.

**Universal installer (Linux, macOS, Windows).** What the GEM Foundation
recommends. It creates its own virtual environment under ``~/openquake``:

.. code-block:: sh

   curl -L -O https://github.com/gem/oq-engine/raw/master/install.py
   python install.py user

Then activate it:

.. code-block:: sh

   source ~/openquake/bin/activate

on Linux and macOS, or

.. code-block:: powershell

   & "$env:USERPROFILE\openquake\Scripts\activate.ps1"

on Windows.

**pip, into an environment of your own.** Simpler, and enough for reading
datastores:

.. code-block:: powershell

   python -m venv .venv-oq
   .\.venv-oq\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install openquake.engine

or, on Linux and macOS,

.. code-block:: sh

   python -m venv .venv-oq
   source .venv-oq/bin/activate
   python -m pip install --upgrade pip
   pip install openquake.engine

**Windows self-installing package.** To avoid the terminal entirely, the
``.exe`` on the
`engine releases page <https://github.com/gem/oq-engine/releases>`_
installs the engine together with its own Python.

Check the installation, and add ``djura`` to the same environment:

.. code-block:: sh

   oq --version
   pip install djura

Step 1: run the PSHA calculation
--------------------------------

From ``tests/rs/exact/assets``:

.. code-block:: sh

   oq engine --run job.ini

The example job file uses ``calculation_mode = disaggregation``, which runs
the classical hazard calculation and then disaggregates it, so the hazard
curves and the rupture contexts end up in one datastore. A
``maximum_distance`` of 50 km keeps the rupture count, and therefore the
selection, tractable.

The engine writes results into its own data directory, not into the working
directory:

.. code-block:: text

   C:\Users\<your-username>\oqdata\calc_<id>.hdf5
   C:\Users\<your-username>\Documents\oqdata\calc_<id>.hdf5
   ~/oqdata/calc_<id>.hdf5

``OQ_DATADIR`` overrides that location. The calculation ID is printed at
the end of the run and can be looked up again with:

.. code-block:: sh

   oq engine --list-hazard-calculations

The datastore of this example was calculation 9, copied to
``tests/rs/exact/data/calc_9.hdf5``. Adjust the ``dstore`` variable at the
top of the script if yours has a different ID.

Step 2: extract the context with ``djura-tools``
------------------------------------------------

``get_context_from_dstore`` is **not** part of the ``djura`` package. It
lives in the companion repository
`djura-tools <https://github.com/djura-risk-data-engineering/djura-tools>`_,
because it imports the OQ-engine to open the datastore and ``djura`` itself
has no dependency on the engine:

.. code-block:: sh

   git clone https://github.com/djura-risk-data-engineering/djura-tools
   cd djura-tools

Run it from that clone, with the engine environment active, so that
``djura.hazard.dstore`` resolves:

.. code-block:: python

   from djura.hazard.dstore import get_context_from_dstore

   # im_ref is the IM the hazard curve is read in. It does not have to be
   # the IM the OQ disaggregation was conditioned on, and it can be
   # changed afterwards on the returned dictionary.
   ctx, oq = get_context_from_dstore("data/calc_9.hdf5", im_ref="SA(0.5)")

``ctx`` is a plain dictionary, and pickling it is what makes the rest of
the workflow independent of the engine:

.. code-block:: python

   from djura.utilities import export_results

   export_results("data/ctx", ctx, "pickle")

What the context holds
~~~~~~~~~~~~~~~~~~~~~~

These are the keys :class:`~djura.record_selection.gcim.GCIM` reads, and
they are worth knowing because they are what the exact approach buys over
the approximate one:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Key
     - Meaning
   * - ``ctx_by_grp``
     - Source-model ID to a numpy record array of rupture contexts. One row
       per rupture, holding every parameter the GMMs require. This is the
       ensemble that becomes ``ruptures``
   * - ``required-parameters``
     - The rupture and distance parameters the GMM logic tree needs, so
       only those are carried over from each row
   * - ``site-parameters``
     - Site parameters of the run, taken from the site model rather than
       re-entered by hand
   * - ``hazard-curves``, ``imt``, ``oq-poes``
     - The hazard curves, the IM levels per IM, and the PoEs of the run.
       Together they give the conditioning value at a requested PoE
   * - ``lt-weights``, ``gsims``, ``gsim-weights``, ``data``
     - The GMM logic tree: leaf weights, being the GMM weight times the
       source-model weight, and the GMM names per source model
   * - ``invtime``, ``phi_b``, ``totrups``
     - Investigation time, the truncation-level probability, and the total
       rupture count
   * - ``im_ref``
     - Which IM the conditioning value is read in. Assign to it to condition
       on a different IM without re-reading the datastore

.. _exact-selection:

Step 3: select the records
--------------------------

From here on the OQ-engine is no longer involved: the pickle is enough, and
``pip install djura`` is the only requirement.

.. code-block:: python

   import pickle

   from djura.record_selection.gcim import GCIM
   from djura.utilities import export_results

   with open("data/ctx.pickle", "rb") as f:
       oq_data = pickle.load(f)

   # Condition on whichever IM of the run records are wanted for
   oq_data["im_ref"] = "SA(1.0)"

   gcim = GCIM("assets/default_data.json", conditional=True,
               dis_oq=oq_data, poe_for_selection=0.02)

   target = gcim.create()
   records = gcim.select()

   export_results("target_0.02", target, "json")
   export_results("records_0.02", records, "json")

``poe_for_selection`` normally is one of the PoEs of the run, that is, one
of ``ctx['oq-poes']``. A value in between is accepted, and the conditioning
intensity is then averaged between the two bracketing levels; a value
outside the range raises.

When ``dis_oq`` is given, the entries ``gmms``, ``site-parameters``,
``ruptures``, ``imi`` and ``im-star`` of the input are **overridden** by the
datastore. That is why ``assets/default_data.json`` carries the selection
settings only:

.. code-block:: json

   {
       "num-components": 2,
       "component-definition": "RotD50",
       "nreplicate": 1,
       "num_records": 40,
       "context_limits": {},
       "seed": 0,
       "ks_alpha": 0.05,
       "im_weights": [],
       "max_scaling_factor": 3.0
   }

The IM vector ``imi`` becomes the intensity measures of the job file, so the
target is built over exactly the IMs the PSHA produced hazard curves for.

Keeping the script in step with the job file
--------------------------------------------

The conditioning IMs and the PoEs are not repeated in the script. They are
read from ``job.ini``, so a change to the PSHA run cannot leave the
selection describing a hazard model that no longer exists:

.. code-block:: python

   import re
   from configparser import RawConfigParser


   def read_job_ini(job_ini="assets/job.ini"):
       parser = RawConfigParser()
       parser.read(job_ini, encoding="utf-8")

       imtls = parser["calculation"]["intensity_measure_types_and_levels"]
       # Keys of the dict literal, i.e. '"PGA": logscale(...)' -> PGA
       im_refs = re.findall(r"""["']([^"']+)["']\s*:""", imtls)

       poes = [float(poe)
               for poe in parser["output"]["poes"].replace(",", " ").split()]

       return im_refs, poes


   im_refs, poes = read_job_ini()

For the example job file this yields 46 IMs, from ``PGA`` through the
spectral accelerations, ``Sa_avg2``, ``Sa_avg3``, ``PGV`` and ``FIV3``, at
the 12 PoEs from 0.4 down to 0.0001. ``RawConfigParser`` is used so that
the ``logscale(...)`` values pass through untouched.

Running the whole case
----------------------

``exact_rs.py`` builds the context and then loops the selection over every
IM and PoE, one process per IM:

.. code-block:: sh

   python tests/rs/exact/exact_rs.py

Results are written per conditioning IM and PoE:

.. code-block:: text

   tests/rs/exact/data/records/<im_ref>/target_<poe>.json
   tests/rs/exact/data/records/<im_ref>/records_<poe>.json

Two practical notes. All 46 IMs at 12 PoEs is 552 selections against the
full rupture ensemble, which is hours rather than minutes, so trim
``im_refs`` to the IMs actually needed. And each worker process holds its
own copy of the record metadata, so lower the pool size if memory is tight;
the script drops each ``GCIM`` instance and collects it before the next PoE
for the same reason.

The script is deliberately not a pytest module: it depends on an optional
OQ-engine installation, and it takes far too long for a test run.

Troubleshooting
---------------

``ModuleNotFoundError: djura.hazard.dstore``
   The datastore helpers are in
   `djura-tools <https://github.com/djura-risk-data-engineering/djura-tools>`_,
   not in the installed ``djura`` wheel. Run the script from a clone of that
   repository.

``ModuleNotFoundError: openquake``
   ``get_context_from_dstore`` opens the datastore through the engine.
   Activate the environment the engine was installed into.

The datastore cannot be read, or its version is refused
   A datastore is tied to the engine version that wrote it. Read it with the
   same version, or re-run the calculation with the version at hand.

``ValueError`` on the PoE
   ``poe_for_selection`` has to lie within ``ctx['oq-poes']``, which is the
   ``poes`` line of the job file. Reading the PoEs from ``job.ini``, as
   above, avoids this.

``KeyError`` on ``im_ref``
   The conditioning IM must be one of the job file's
   ``intensity_measure_types_and_levels``. The engine only stores hazard
   curves for the IMs it was asked for.
