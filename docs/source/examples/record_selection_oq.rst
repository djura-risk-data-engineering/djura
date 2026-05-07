Record selection from OpenQuake PSHA disaggregation
====================================================

This example shows how to drive GCIM-based record selection directly
from an `OpenQuake Engine <https://github.com/gem/oq-engine>`_ PSHA
disaggregation output. In this workflow the rupture scenarios and the
conditioning IM value are not defined by hand. They are read from the
OQ disaggregation context so that the selection is fully consistent
with your site-specific hazard model.

When ``dis_oq`` is supplied, the ``"ruptures"`` and ``"im-star"``
entries in the base input file are overridden automatically by the
disaggregation data.

Prerequisites
-------------

You need two files produced by an OQ PSHA run:

* **Base input JSON**: defines the GMM ensemble, site parameters, IM
  list, and selection settings (same format as the standalone example).
  Download: `input.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/assets/oq_dis/input.json>`_
* **Context pickle**: the preprocessed OQ disaggregation context
  (``ctx_<poe>_<imt>.pickle``), produced by djura's OQ post-processing
  utilities. Downloads:
  `ctx_80_AvgSA(0.5).pickle <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/assets/oq_dis/ctx_80_AvgSA(0.5).pickle>`_,
  `ctx.pickle <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/assets/oq_dis/ctx.pickle>`_

Base input file
---------------

.. code-block:: json

   {
     "gmms": [
       {
         "Sa_avg": {
           "names": ["KothaEtAl2020ESHM20"],
           "weights": [1.0]
         }
       }
     ],
     "site-parameters": {
       "vs30": 500,
       "z2pt5": 2,
       "soil": 0,
       "mechanism": "normal fault",
       "backarc": 0,
       "xvf": 10,
       "region": 0
     },
     "ruptures": [
       {
         "mag": 6.3, "rjb": 10, "weight": 0.6,
         "rrup": 12, "d_hyp": 10, "rx": 120, "ztor": 0, "rhypo": 10
       },
       {
         "mag": 6.0, "rjb": 20, "weight": 0.4,
         "rrup": 20, "d_hyp": 5,  "rx": 120, "ztor": 0, "rhypo": 10
       }
     ],
     "imi": [
       "Sa_avg(0.05s)", "Sa_avg(0.075s)", "Sa_avg(0.1s)",
       "Sa_avg(0.15s)", "Sa_avg(0.2s)",  "Sa_avg(0.25s)",
       "Sa_avg(0.3s)",  "Sa_avg(0.4s)",  "Sa_avg(0.5s)",
       "Sa_avg(0.75s)", "Sa_avg(1.0s)",  "Sa_avg(1.3s)",
       "Sa_avg(1.5s)",  "Sa_avg(2.0s)"
     ],
     "im-star": {
       "type": "Sa_avg(0.5s)",
       "value": 0.3,
       "gmms": {
         "names": ["KothaEtAl2020ESHM20"],
         "weights": [1.0]
       }
     },
     "nreplicate": 1,
     "num_records": 40,
     "context_limits": {},
     "seed": 0,
     "ks_alpha": 0.05,
     "im_weights": [],
     "max_scaling_factor": 3
   }

The ``"ruptures"`` and ``"im-star"`` blocks above act as fallbacks
only. They are replaced at runtime by the disaggregation data.

Running the selection
---------------------

.. code-block:: python

   import json
   import pickle
   from djura.record_selection import GCIM

   # Load the base input (GMM ensemble, site params, selection settings)
   with open("input.json", encoding="utf-8") as f:
       base_input = json.load(f)

   # Load the preprocessed OQ disaggregation context.
   # The filename encodes the PoE and conditioning IM used in the OQ run.
   # Example: ctx_80_AvgSA(0.5).pickle → 10 % PoE in 50 yr, Sa_avg at 0.5 s
   with open("ctx_80_AvgSA(0.5).pickle", "rb") as f:
       oq_data = pickle.load(f)

   # poe_for_selection picks the hazard level from the disaggregation output.
   # 0.1 corresponds to 10 % probability of exceedance in 50 years.
   gcim = GCIM(base_input, conditional=True,
               dis_oq=oq_data, poe_for_selection=0.1)

   # Step 1: build GCIM distributions using the OQ rupture ensemble.
   gcim.create()

   # Step 2: select records from the NGA-West2 database.
   gcim.select()

What the OQ context overrides
------------------------------

When ``dis_oq`` is provided, djura replaces the following fields from
the base input with values derived from the disaggregation:

* **ruptures**: the full logic-tree rupture ensemble (source model x
  GMM branches), weighted by their disaggregation contribution.
* **im-star value**: the hazard-consistent IM level at the requested
  PoE, read directly from the OQ hazard curve.
* **im-star GMMs**: reweighted to match the OQ logic-tree branches
  that contributed to the disaggregation at that PoE.

Everything else (site parameters, IM list, selection settings) is
taken from the base input file unchanged.

Using a raw context pickle (``ctx.pickle``)
-------------------------------------------

If you have a generic context pickle without an embedded PoE, set
``im_ref`` on the loaded object and pass a PoE explicitly:

.. code-block:: python

   import json
   import pickle
   from djura.record_selection import GCIM

   with open("input.json", encoding="utf-8") as f:
       base_input = json.load(f)

   with open("ctx.pickle", "rb") as f:
       oq_data = pickle.load(f)

   # Specify which IM the hazard curve is expressed in
   oq_data["im_ref"] = "SA(0.5)"

   gcim = GCIM(base_input, conditional=True,
               dis_oq=oq_data, poe_for_selection=0.0025)
   gcim.create()
