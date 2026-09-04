Example 1: A trivial logic tree case
====================================

The seismic source model is ZS9, proposed by Meletti et al. (2008) and
adopted for the seismic hazard assessment of Italy (Stucchi et al. 2009,
2011). The ground-motion logic tree collapses to a single branch, defined
using the GMM of Aristeidou et al. (2024), developed for active shallow
crustal regions.

Both branch sets therefore carry weight 1, and the tree resolves to a single
realization over a single source group. None of the logic-tree machinery
:doc:`example 2 <example2>` exercises is present here, which is what makes
this the one to read first: the run is wide exactly where its tree is
narrow, at 46 intensity measures and 12 probabilities of exceedance in 50
years for a site in Rivisondoli, Italy, taken from a site model.

Files of the example
---------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path under ``tests/rs/exact``
     - What it is
   * - ``assets/job_1.ini``
     - The OQ job file. A disaggregation run for a site in Rivisondoli,
       Italy, over 46 intensity measures and 12 probabilities of exceedance
       in 50 years
   * - ``assets/inputs_1/``
     - The source-model logic tree (``ssmLT.xml``) over the single ZS9
       branch in ``ssm.xml``, the GMM logic tree (``gmmLT.xml``) over the
       single ``AristeidouEtAl2024`` branch, and the site models
       (``rivisondoli.csv``, ``laquila.csv``) referenced by ``job_1.ini``
   * - ``assets/default_data.json``
     - The selection settings only, shared by both examples: number of
       records, seed, KS level, maximum scaling factor, component
       definition. The hazard entries are absent on purpose, as they come
       from the datastore
   * - ``data/calc_1.hdf5``
     - The datastore of that run, copied out of ``oqdata``
   * - ``data/ctx_1.pickle``
     - The context extracted from the datastore, so the selection can be
       re-run without the OQ-engine installed

Step 1: run the PSHA calculation
---------------------------------

From ``tests/rs/exact/assets``:

.. code-block:: sh

   oq engine --run job_1.ini

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
``tests/rs/exact/data/calc_1.hdf5``.

Step 2: extract the context with ``djura-tools``
-------------------------------------------------

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
   ctx, oq = get_context_from_dstore("data/calc_1.hdf5", im_ref="SA(0.5)")

``ctx`` is a plain dictionary, and pickling it is what makes the rest of
the workflow independent of the engine:

.. code-block:: python

   from djura.utilities import export_results

   export_results("data/ctx_1", ctx, "pickle")

.. _context-keys:

What the context holds
~~~~~~~~~~~~~~~~~~~~~~~

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
---------------------------

From here on the OQ-engine is no longer involved: the pickle is enough, and
``pip install djura`` is the only requirement.

.. code-block:: python

   import pickle

   from djura.record_selection.gcim import GCIM
   from djura.utilities import export_results

   with open("data/ctx_1.pickle", "rb") as f:
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
---------------------------------------------

The conditioning IMs and the PoEs are not repeated in the script. They are
read from ``job_1.ini``, so a change to the PSHA run cannot leave the
selection describing a hazard model that no longer exists:

.. code-block:: python

   import re
   from configparser import RawConfigParser


   def read_job_ini(job_ini="assets/job_1.ini"):
       parser = RawConfigParser()
       parser.read(job_ini, encoding="utf-8")

       imtls = parser["calculation"]["intensity_measure_types_and_levels"]
       # Keys of the dict literal, i.e. '"PGA": logscale(...)' -> PGA
       im_refs = re.findall(r"""["']([^"']+)["']\s*:""", imtls)

       poes = [float(poe)
               for poe in parser["output"]["poes"].replace(",", " ").split()]

       return im_refs, poes


   im_refs, poes = read_job_ini()

For this example job file this yields 46 IMs, from ``PGA`` through the
spectral accelerations, ``Sa_avg2``, ``Sa_avg3``, ``PGV`` and ``FIV3``, at
the 12 PoEs from 0.4 down to 0.0001. ``RawConfigParser`` is used so that
the ``logscale(...)`` values pass through untouched.

Running the whole case
-----------------------

With ``dstore = "1"`` set at the top of ``exact_rs.py``, it builds the
context and then loops the selection over every IM and PoE, one process
per IM:

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
