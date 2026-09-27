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
   from djura import im_conversion
   from djura import signal_processing

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

   # the bundled dataset is downloaded and cached automatically on first use
   gcim = GCIM(data="path/to/input.json", conditional=True)
   gcim.create()
   gcim.select()

Intensity measures of a record
------------------------------

.. code-block:: python

   from djura import signal_processing as sig

   dt, npts, desc, t, acc1 = sig.read_nga("RSN179_IMPVALL.H_H-E04140.AT2")
   _, _, _, _, acc2 = sig.read_nga("RSN179_IMPVALL.H_H-E04230.AT2")

   pair = sig.GroundMotion(acc1, acc2, dt, unit="g")
   sa50 = pair.rotd("psa", 50, [0.2, 1.0, 2.0])   # RotD50 spectrum, g
   pair.first.ia                                   # Arias intensity, m/s
