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

   * :doc:`../record_selection` selects records from a hazard input written by
     hand, with ``djura`` alone. Nothing else has to be installed.
   * :doc:`../record_selection_oq` drives the selection from an OQ
     disaggregation in the *approximate* form: the exported disaggregation
     is read as magnitude-distance bins, and each bin becomes one
     representative scenario.
   * This page is the *exact* form of that same workflow, for when the
     binning is the thing to avoid.

Approximate against exact
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Approximate (:doc:`../record_selection_oq`)
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

Step 0: install the OQ-engine
-------------------------------

Needed only to read a datastore, that is, only for ``build_context()``. If
you were given a ``ctx_<n>.pickle``, skip to :ref:`exact-selection`.

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

Examples
--------

Two example PSHA runs are bundled under ``tests/rs/exact``, each set apart
by a numeric suffix ``<n>`` (``1`` or ``2``): ``assets/job_<n>.ini``,
``assets/inputs_<n>/``, ``data/calc_<n>.hdf5`` and ``data/ctx_<n>.pickle``.
Pick one by setting the ``dstore`` variable near the top of ``exact_rs.py``
to ``"1"`` or ``"2"``; everything else in the script, including
``read_job_ini()``'s default and the pickled context path, follows from it.
``dstore`` is this local example number, not the OQ-engine's own
calculation id — after running the engine, its output is copied from
``oqdata`` into ``tests/rs/exact/data/calc_<n>.hdf5`` under the example
number, whatever the engine happened to call it.

.. toctree::
   :maxdepth: 1

   example1
   example2

The two pull in opposite directions on purpose. Example 1 is a realistic
hazard study, wide in intensity measures and hazard levels but with a
logic tree of a single branch. Example 2 is the mirror image: two IMs' worth
of breadth, one hazard level, and a logic tree of twelve realizations across
two tectonic region types. Read the first for the workflow, the second for
what the exact approach does with logic-tree weights.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - :doc:`Example 1 <example1>`
     - :doc:`Example 2 <example2>`
   * - Site
     - Rivisondoli, Italy, from a site model
     - ``0.5 -0.5``, with ``vs30 = 600`` m/s set in the job file
   * - Source-model logic tree
     - One branch: ZS9, after Meletti et al. (2008)
     - Three branches, weighted 0.5 / 0.3 / 0.2
   * - GMM logic tree
     - One branch: Aristeidou et al. (2024), active shallow crust
     - Two branches on each of two tectonic region types
   * - Realizations
     - 1
     - 12, fully enumerated
   * - Source groups
     - 1
     - 6, holding 23 016 ruptures
   * - IMs and PoEs
     - 46 IMs at 12 PoEs, so 552 selections
     - 22 IMs at 1 PoE, so 22 selections
   * - ``maximum_distance``
     - 50 km
     - 200 km

Troubleshooting
----------------

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
   ``poes`` line of the job file. Reading the PoEs from ``job_<n>.ini``, as
   above, avoids this.

``KeyError`` on ``im_ref``
   The conditioning IM must be one of the job file's
   ``intensity_measure_types_and_levels``. The engine only stores hazard
   curves for the IMs it was asked for.
