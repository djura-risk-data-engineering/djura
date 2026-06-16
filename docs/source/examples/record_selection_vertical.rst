Vertical component record selection
===================================

This example extends the conditional GCIM-based record selection to
account for the **vertical** spectral acceleration component
(``SA_vert``) alongside the horizontal one. A dedicated vertical GMM is
supplied for ``SA_vert`` and vertical IMs are added to the target IM
list, so the selection matches both horizontal and vertical spectra.

Input file
----------

Download:
`input17.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/assets/gcim_inputs/input17.json>`_

The selection is driven by a JSON input file. The one used here
specifies one horizontal GMM for ``SA`` (``AristeidouEtAl2024``) and one
vertical GMM for ``SA_vert`` (``GulerceEtAl2017``), a single rupture
scenario, a conditioning IM of SA(1.0 s) = 0.3 g, and a target of 40
records with a maximum scaling factor of 3. The ``imi`` list includes
both horizontal ``SA(T)`` and vertical ``SA_vert(T)`` ordinates, and
``num_components`` is set to 3 so the vertical component is selected
together with the two horizontal components.

.. code-block:: json

   {
     "gmms": [
       {
         "SA": {
           "names": ["AristeidouEtAl2024"],
           "weights": [1.0]
         }
       },
       {
         "SA_vert": {
           "names": ["GulerceEtAl2017"],
           "weights": [1.0]
         }
       }
     ],
     "site-parameters": {
       "vs30": 500,
       "z2pt5": 2,
       "soil": 0,
       "mechanism": "normal fault"
     },
     "ruptures": [
       {
         "mag": 6.3, "rjb": 10, "weight": 0.6,
         "rrup": 12, "d_hyp": 10, "rx": 120, "ztor": 0
       }
     ],
     "imi": [
       "SA(0.05s)", "SA(0.075s)", "SA(0.1s)", "SA(0.15s)", "SA(0.2s)",
       "SA(0.25s)", "SA(0.3s)", "SA(0.4s)", "SA(0.5s)", "SA(0.75s)",
       "SA(1.0s)", "SA(1.3s)", "SA(1.5s)", "SA(2.0s)",
       "SA_vert(0.05s)", "SA_vert(0.075s)", "SA_vert(0.1s)",
       "SA_vert(0.15s)", "SA_vert(0.2s)", "SA_vert(0.25s)",
       "SA_vert(0.3s)", "SA_vert(0.4s)", "SA_vert(0.5s)",
       "SA_vert(0.75s)", "SA_vert(1.0s)", "SA_vert(1.3s)",
       "SA_vert(1.5s)", "SA_vert(1.55s)", "SA_vert(2.0s)"
     ],
     "im-star": {
       "type": "SA(1.)",
       "value": 0.3
     },
     "num_components": 3,
     "nreplicate": 1,
     "num_records": 40,
     "context_limits": {},
     "seed": 0,
     "ks_alpha": 0.05,
     "im_weights": [],
     "max_scaling_factor": 3
   }

Running the selection
---------------------

Save the JSON above as ``input.json``, then:

.. code-block:: python

   from djura.record_selection import GCIM

   # Pass the path to your input file and enable conditional selection.
   # The NGA-West2 dataset is downloaded and cached automatically on first use.
   gcim = GCIM("input.json", conditional=True)

   # Step 1: build the GCIM distributions for all IMs in "imi",
   #         including the vertical SA_vert ordinates.
   gcim.create()

   # Step 2: select records matching both horizontal and vertical targets.
   gcim.select()

After ``select()`` completes, the chosen record IDs, scaling factors,
and goodness-of-fit statistics — for the horizontal and vertical
components alike — are available on the ``gcim`` object.

.. note::

   Vertical-component support requires a vertical GMM (such as
   ``GulerceEtAl2017``) to be supplied under the ``SA_vert`` key in
   ``gmms``, and the corresponding ``SA_vert(T)`` entries to appear in
   ``imi``. See :doc:`record_selection` for the horizontal-only
   workflow.
