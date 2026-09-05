Example 2: A non-trivial logic tree case
=========================================

``job_2.ini`` describes itself as a *disaggregation with non-trivial logic
tree*, and that is the point of it: three source models, two GMMs on each of
two tectonic region types, enumerated in full. Where
:doc:`example 1 <example1>` resolves to a
single realization, this one has twelve, each carrying its own weight into
the GCIM target, so it is the case to read when the question is how
logic-tree weights survive the trip from the datastore into the selection.

The site is synthetic, at longitude 0.5 and latitude -0.5, and the site
parameters are reference values in the job file rather than a site model:

.. code-block:: ini

   [site_params]
   reference_vs30_type = measured
   reference_vs30_value = 600.0
   reference_depth_to_2pt5km_per_sec = 5.0
   reference_depth_to_1pt0km_per_sec = 100.0

The IM set is 22 entries — ``PGA``, ``SA(0.05)``, and ``SA(0.1)`` through
``SA(2.0)`` in steps of 0.1 — against a single PoE of ``0.1``. That is 22
selections rather than example 1's 552, which is what makes this the one to
run first.

Files of the example
----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path under ``tests/rs/exact``
     - What it is
   * - ``assets/job_2.ini``
     - The OQ job file. A disaggregation run for a synthetic site over 22
       intensity measures at one probability of exceedance in 50 years, with
       ``maximum_distance`` at 200 km
   * - ``assets/inputs_2/``
     - ``source_model_logic_tree.xml`` with its three source models
       (``source_model_1.xml`` to ``source_model_3.xml``), and
       ``gmpe_logic_tree.xml``
   * - ``assets/default_data.json``
     - The same selection settings example 1 uses; nothing in it is
       example-specific
   * - ``data/calc_2.hdf5``
     - The datastore of that run, copied out of ``oqdata``
   * - ``data/ctx_2.pickle``
     - The context extracted from the datastore

The logic tree
----------------

One tree per tectonic setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source-model logic tree is *global*: it swaps the entire source model,
so it sits above every tectonic region type.

.. list-table::
   :header-rows: 1
   :widths: 16 50 34

   * - Branch
     - Source model
     - Weight
   * - ``b1``
     - ``source_model_1.xml``
     - 0.5
   * - ``b2``
     - ``source_model_2.xml``
     - 0.3
   * - ``b3``
     - ``source_model_3.xml``
     - 0.2

The GMM logic tree is *not* global. Each ``logicTreeBranchSet`` carries an
``applyToTectonicRegionType``, so it only ever touches sources tagged with
that TRT. The engine also renames the branches: the ids declared in the XML
never appear in the outputs, and the realization paths use the renamed ones
instead — a letter per branch, a digit per TRT index.

.. list-table::
   :header-rows: 1
   :widths: 12 30 12 12 22 12

   * - Set
     - Tectonic region type
     - XML
     - Engine
     - GMM
     - Weight
   * - ``bs1``
     - Active Shallow Crust
     - ``b11``
     - ``gA0``
     - ``BooreAtkinson2008``
     - 0.6
   * - ``bs1``
     - Active Shallow Crust
     - ``b12``
     - ``gB0``
     - ``ChiouYoungs2008``
     - 0.4
   * - ``bs2``
     - Stable Continental Crust
     - ``b21``
     - ``gA1``
     - ``AtkinsonBoore2006``
     - 0.7
   * - ``bs2``
     - Stable Continental Crust
     - ``b22``
     - ``gB1``
     - ``Atkinson2008prime``
     - 0.3

The two settings share no source, so each one effectively walks its own
two-level tree of six paths. The fault sources of the Active Shallow group
only ever see ``BooreAtkinson2008`` or ``ChiouYoungs2008``; which Stable
Continental GMM the run happens to be using is invisible to them. The area
sources of the Stable Continental group only ever see ``AtkinsonBoore2006``
or ``Atkinson2008prime``, and at different weights, 0.7 / 0.3 against
0.6 / 0.4.

The full enumerated tree
~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``number_of_logic_tree_samples = 0``, the engine enumerates every
path instead of sampling. It composes the branch sets in file order, source
model then Active Shallow then Stable Continental, with the last varying
fastest. That gives 3 x 2 x 2 = 12 realizations, each weighted by the
product of its branch weights. The path is the source-model path, a tilde,
then one GMM branch id per TRT in TRT order:

.. list-table::
   :header-rows: 1
   :widths: 26 42 32

   * - Realization
     - Path
     - Weight
   * - ``rlz0``
     - ``b1~gA0gA1``
     - 0.210
   * - ``rlz1``
     - ``b1~gA0gB1``
     - 0.090
   * - ``rlz2``
     - ``b1~gB0gA1``
     - 0.140
   * - ``rlz3``
     - ``b1~gB0gB1``
     - 0.060
   * - ``rlz4``
     - ``b2~gA0gA1``
     - 0.126
   * - ``rlz5``
     - ``b2~gA0gB1``
     - 0.054
   * - ``rlz6``
     - ``b2~gB0gA1``
     - 0.084
   * - ``rlz7``
     - ``b2~gB0gB1``
     - 0.036
   * - ``rlz8``
     - ``b3~gA0gA1``
     - 0.084
   * - ``rlz9``
     - ``b3~gA0gB1``
     - 0.036
   * - ``rlz10``
     - ``b3~gB0gA1``
     - 0.056
   * - ``rlz11``
     - ``b3~gB0gB1``
     - 0.024

That column is ``ctx['lt-weights']``, in that order: entry ``i`` is the
weight of realization ``i``. The modal branch is ``b1~gA0gA1`` at
0.5 x 0.6 x 0.7 = 0.210 and the least likely is ``b3~gB0gB1`` at
0.2 x 0.4 x 0.3 = 0.024, a range of 8.75x across the tree. The weights sum
to 1.000, with 0.5 spread across the four ``b1`` leaves, 0.3 across ``b2``
and 0.2 across ``b3``.

The engine never computes a realization end to end. It partitions the
sources into one *source group* per source model and TRT pair, six here, and
computes rates for each group under each GMM that group can see. That is
6 groups x 2 GMMs = 12 rate sets. Those six groups are the keys of
``ctx['ctx_by_grp']``:

.. list-table::
   :header-rows: 1
   :widths: 8 24 14 14 26 14

   * - Group
     - Tectonic region type
     - Sources
     - Ruptures
     - GMMs in group
     - Serves
   * - ``0``
     - Active Shallow Crust
     - ``2!b1``
     - 1 334
     - ``BooreAtkinson2008`` / ``ChiouYoungs2008``
     - 0,1 / 2,3
   * - ``1``
     - Active Shallow Crust
     - ``2!b2``
     - 1 297
     - ``BooreAtkinson2008`` / ``ChiouYoungs2008``
     - 4,5 / 6,7
   * - ``2``
     - Active Shallow Crust
     - ``2!b3`` + ``3``
     - 1 334 + 831
     - ``BooreAtkinson2008`` / ``ChiouYoungs2008``
     - 8,9 / 10,11
   * - ``3``
     - Stable Continental Crust
     - ``1!b1``
     - 4 100
     - ``AtkinsonBoore2006`` / ``Atkinson2008prime``
     - 0,2 / 1,3
   * - ``4``
     - Stable Continental Crust
     - ``1!b2``
     - 5 920
     - ``AtkinsonBoore2006`` / ``Atkinson2008prime``
     - 4,6 / 5,7
   * - ``5``
     - Stable Continental Crust
     - ``1!b3`` + ``4``
     - 4 100 + 4 100
     - ``AtkinsonBoore2006`` / ``Atkinson2008prime``
     - 8,10 / 9,11

That is 23 016 ruptures in total, which is ``ctx['totrups']``, and the
twelve rate sets are the twelve entries of ``ctx['gsims']`` and
``ctx['gsim-weights']``.

Step 1: run the PSHA calculation
----------------------------------

From ``tests/rs/exact/assets``:

.. code-block:: sh

   oq engine --run job_2.ini

Copy the resulting datastore out of ``oqdata`` to
``tests/rs/exact/data/calc_2.hdf5``, as in example 1.

Step 2: extract the context
-----------------------------

Identical to :doc:`example 1 <example1>`, against the other datastore:

.. code-block:: python

   from djura.hazard.dstore import get_context_from_dstore
   from djura.utilities import export_results

   ctx, oq = get_context_from_dstore("data/calc_2.hdf5", im_ref="SA(0.5)")
   export_results("data/ctx_2", ctx, "pickle")

The keys are the ones tabulated under
:ref:`What the context holds <context-keys>`; what differs is how much they
carry. ``ctx['ctx_by_grp']`` has six entries rather than one,
``ctx['gsims']`` and ``ctx['gsim-weights']`` twelve rather than one, and
``ctx['site-parameters']`` is ``['vs30', 'vs30measured', 'z1pt0']``, the
parameters the four GMMs of this tree require between them.

Step 3: select the records
----------------------------

The call is the one from example 1, with the other pickle and the single PoE
of the run:

.. code-block:: python

   import pickle

   from djura.record_selection.gcim import GCIM

   with open("data/ctx_2.pickle", "rb") as f:
       oq_data = pickle.load(f)

   oq_data["im_ref"] = "SA(0.5)"

   gcim = GCIM("assets/default_data.json", conditional=True,
               dis_oq=oq_data, poe_for_selection=0.1)

   target = gcim.create()
   records = gcim.select()

``poe_for_selection`` has to be ``0.1`` here: it is the only entry of
``ctx['oq-poes']``, because ``job_2.ini`` asks for one. Every rupture of all
six groups enters the target, weighted by its logic-tree leaf weight and its
rate of occurrence.

Running the whole case
------------------------

Set ``dstore = "2"`` at the top of ``exact_rs.py`` and run it as before:

.. code-block:: sh

   python tests/rs/exact/exact_rs.py

``read_job_ini()`` then reads ``assets/job_2.ini`` and yields the 22 IMs and
the single PoE, so the run is 22 selections. Note that results are written
to ``data/records/<im_ref>/`` without the example number in the path, so
move or clear that directory between examples if you want to keep both.
