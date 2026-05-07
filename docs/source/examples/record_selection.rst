Ground motion record selection
==============================

This example walks through a conditional GCIM-based record selection
using a logic-tree ground motion model (GMM) ensemble.

Input file
----------

The selection is driven by a JSON input file. The one used here
specifies two GMMs for SA (weighted 40 %/60 %), one GMM for Ds595,
two rupture scenarios, a conditioning IM of SA(1.0 s) = 0.7 g, and a
target of 40 records with a maximum scaling factor of 3.

.. code-block:: json

   {
     "gmms": [
       {
         "SA": {
           "names": ["AristeidouEtAl2024", "BooreEtAl2020"],
           "weights": [0.4, 0.6]
         }
       },
       {
         "Ds595": {
           "names": ["AbrahamsonSilva1996"],
           "weights": [1]
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
       },
       {
         "mag": 6.0, "rjb": 20, "weight": 0.4,
         "rrup": 20, "d_hyp": 5,  "rx": 120, "ztor": 0
       }
     ],
     "imi": [
       "SA(0.05s)", "SA(0.075s)", "SA(0.1s)", "SA(0.15s)", "SA(0.2s)",
       "SA(0.25s)", "SA(0.3s)",  "SA(0.4s)", "SA(0.5s)", "SA(0.75s)",
       "SA(1.0s)",  "SA(1.3s)",  "SA(1.5s)", "SA(2.0s)", "Ds595"
     ],
     "im-star": {
       "type": "SA(1.)",
       "value": 0.7,
       "gmms": {
         "names": ["BooreEtAl2020", "AristeidouEtAl2024"],
         "weights": [0.4, 0.6]
       }
     },
     "num-components": 1,
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

   # Step 1: build the GCIM distributions for all IMs in "imi".
   gcim.create()

   # Step 2: select records from the NGA-West2 database.
   gcim.select()

After ``select()`` completes, the chosen record IDs, scaling factors,
and goodness-of-fit statistics are available on the ``gcim`` object.

Discovering supported options
------------------------------

``GCIM`` exposes several helper methods that return the parameters and
models available in the current installation:

.. code-block:: python

   from djura.record_selection import GCIM

   gcim = GCIM()

   # Rupture, site, and distance parameters accepted by the GMMs
   print(gcim.get_supported_rupture_parameters())
   print(gcim.get_supported_sites_parameters())
   print(gcim.get_supported_distances_parameters())

   # NGA-West2 metadata column names
   print(gcim.get_metadata_parameters())

   # Correlation models and IM component conventions
   print(gcim.available_correlation_models())
   print(gcim.get_supported_im_component_types())

   # All ground motion models bundled with djura
   print(gcim.get_available_gsims())
