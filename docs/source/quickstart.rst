Quickstart
==========

Basic usage
-----------

.. code-block:: python

   import djura

   print(djura.__version__)

Submodule imports
-----------------

.. code-block:: python

   from djura import record_selection
   from djura import hazard_consistency
   from djura import edp_im
   from djura import vulnerability_modeller
   from djura import slf
   from djura import fragility_converter

Citations
---------

.. code-block:: python

   # Umbrella package citation
   print(djura.cite())

   # Per-submodule citation
   print(djura.cite("record_selection"))
   print(djura.cite("vulnerability_modeller"))

   # All citations at once
   print(djura.cite(all=True))

Ground motion record selection
------------------------------

.. code-block:: python

   from djura.record_selection import GCIM

   # NGA-West2 dataset is downloaded and cached automatically on first use
   gcim = GCIM(data="path/to/input.json", conditional=True)
   gcim.create()
   gcim.select()
