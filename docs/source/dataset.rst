Bundled dataset (NGA-West2)
===========================

The NGA-West2 metadata (~107 MB uncompressed) is too large to ship inside
the wheel. It is hosted as a gzip-compressed asset on a GitHub Release and
downloaded automatically the first time it is needed.

Automatic download
------------------

No action is required. The first call to any API that needs the dataset
(e.g. :class:`~djura.record_selection.gcim.GCIM`) triggers a one-time
download:

.. code-block:: python

   from djura.record_selection import GCIM

   gcim = GCIM()   # downloads ~85 MB to ~/.cache/djura/ on first call

Subsequent calls load directly from the local cache at
``~/.cache/djura/NGA_W2_v2.pickle``.

Manual control
--------------

.. code-block:: python

   from djura.data_loader import load_data, clear_cache

   data = load_data()   # explicit load / download
   clear_cache()        # remove cached file to force re-download

Custom dataset path
-------------------

Set the ``DJURA_METADATA_PATH`` environment variable to point to a local
pickle file.  This bypasses the download entirely and is the recommended
approach for offline use or custom metadata:

.. code-block:: bash

   # bash / macOS / Linux
   export DJURA_METADATA_PATH=/path/to/metafile.pickle

   # PowerShell
   $env:DJURA_METADATA_PATH = "C:\path\to\metafile.pickle"

   # CMD
   set DJURA_METADATA_PATH=C:\path\to\metafile.pickle

Attribution
-----------

The dataset is derived from the
`NGA-West2 Ground Motion Database <https://ngawest2.berkeley.edu>`_
(PEER, UC Berkeley). It contains metadata only — no waveform records.
See
`ATTRIBUTION.md <https://github.com/djura-risk-data-engineering/djura/blob/main/src/djura/record_selection/assets/ATTRIBUTION.md>`_
for full attribution and instructions on downloading waveforms from PEER.
