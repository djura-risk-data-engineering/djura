Downloading record time histories
=================================

Record selection returns record identifiers, not waveforms. This example
uses :class:`~djura.record_selection.downloader.ESMDownloader` to fetch
the unscaled acceleration time histories of those records from the
`European Strong Motion database <https://esm-db.eu>`_ and collect them
into a single zipfile.

Prerequisites
-------------

* The ``record_selection`` extra, which provides the HTTP client:

  .. code-block:: bash

      pip install "djura[record_selection]"

* A free ESM account, registered at
  `esm-db.eu <https://esm-db.eu>`_. The web service authenticates every
  download with a signed message ("token") issued for those credentials.

Quick start
-----------

Each record is identified by a pair: the **event ID** and the **station
code** of the recording station.

.. code-block:: python

    import os
    from djura.record_selection import ESMDownloader

    downloader = ESMDownloader(
        download_dir="records",              # created if it does not exist
        events=["INT-20230206_0000008", "INT-20230206_0000008"],
        stations=["1210", "4624"],
        username=os.environ["ESM_USERNAME"],  # your ESM account e-mail
        password=os.environ["ESM_PASSWORD"],
    )

    zip_path = downloader.download()
    print(zip_path)   # records/UnscaledRecords.zip

``events`` and ``stations`` are paired element-wise, so the two lists
must have the same length: the *i*-th record is the *i*-th event as
recorded at the *i*-th station. The same event appears as many times as
it has stations in the request. A mismatch raises ``ValueError``.

Keep credentials out of your source files. Reading them from environment
variables, as above, is the simplest option; during development a
``.env`` file loaded with ``python-dotenv`` works equally well, provided
the file is not committed.

Authentication and the token
----------------------------

``download()`` needs either credentials or an already-issued token:

* **Credentials** (``username`` and ``password``) — a token is requested
  from ESM on the first call and written to ``token_path``, which
  defaults to ``download_dir / "token.txt"``.
* **An existing token** — pass ``token_path`` pointing at a token file
  you already have. When that file exists it is used as is and no
  credentials are needed:

  .. code-block:: python

      downloader = ESMDownloader(
          download_dir="records",
          events=events,
          stations=stations,
          token_path="secrets/token.txt",
      )

If ``token_path`` names a file that does not exist yet, the token is
fetched with the supplied credentials and saved there, so the same call
works on later runs without them. With neither credentials nor an
existing token, ``download()`` raises ``TypeError``.

Treat the token as a credential: it authorises downloads on your account
and belongs outside version control.

Downloading a GCIM selection
----------------------------

After :meth:`~djura.record_selection.gcim.GCIM.select` completes, the
selected records are described by their index into the record metadata,
under the ``Rec_ID`` key. Those indices map to the ESM event IDs and
station names of the bundled flatfile:

.. code-block:: python

    import os
    from djura.data_loader import get_metadata
    from djura.record_selection import GCIM, ESMDownloader

    # Step 1: run the selection (see the record selection example).
    gcim = GCIM("input.json", conditional=True)
    gcim.create()
    records = gcim.select()

    # Step 2: map the selected records back onto the metadata.
    metadata = get_metadata()
    rec_ids = records["selected_scaled_best"]["Rec_ID"]

    events = metadata["esm_event_id"][rec_ids].tolist()

    # Station_name is "<network>-<station>", e.g. "TK-1210"; the web
    # service expects the station code alone.
    stations = [name.split("-", 1)[1]
                for name in metadata["Station_name"][rec_ids]]

    # Step 3: download the waveforms of exactly those records.
    zip_path = ESMDownloader(
        download_dir="records",
        events=events,
        stations=stations,
        username=os.environ["ESM_USERNAME"],
        password=os.environ["ESM_PASSWORD"],
    ).download()

Every record in the flatfile shipped with djura comes from ESM and
carries an ``esm_event_id``, so any selection made against it can be
downloaded this way. A custom flatfile supplied through
``DJURA_METADATA_PATH`` need not follow that convention — records from
another database have no ESM event ID and cannot be retrieved from this
service.

The downloader ignores the scaling factors in
``records["selected_scaled_best"]["sf"]``: what it retrieves are the
unscaled records as recorded. Apply the factors to the downloaded time
histories yourself.

Output
------

The records arrive one per request and are collected into
``download_dir / "UnscaledRecords.zip"``, whose name is available as
``ESMDownloader.ZIPFILE_NAME``. Each record contributes the ASCII
acceleration files served by ESM, named after the network, station,
channel, and event, for example::

    records/
        token.txt
        UnscaledRecords.zip
            TK.1210..HNE.D.INT-20230206_0000008.ACC.MP.ASC
            TK.1210..HNN.D.INT-20230206_0000008.ACC.MP.ASC
            ...

An existing zipfile at that path is overwritten, so download into a
fresh directory per request set if you want to keep earlier results.

The archive is a plain ``zipfile``, so the individual records can be read
without unpacking it to disk:

.. code-block:: python

    from zipfile import ZipFile

    with ZipFile(zip_path) as archive:
        print(archive.namelist())
        contents = archive.read(archive.namelist()[0]).decode()

Progress and errors
-------------------

Progress is reported through the ``logging`` module rather than printed,
so nothing is emitted until logging is configured:

.. code-block:: python

    import logging

    logging.basicConfig(level=logging.INFO)

    # Started executing download method to retrieve selected records ...
    # Token written to records\token.txt
    # Sending download requests...
    # 1/40 of requests are done.
    # ...

Failures surface as exceptions:

* ``ValueError`` — ``events`` and ``stations`` differ in length.
* ``TypeError`` — neither credentials nor an existing token file were
  given.
* ``requests.exceptions.HTTPError`` — ESM rejected the credentials, or a
  record request failed. The message names the event and station that
  failed and the HTTP status code; the response is attached to the
  exception. A rejected token is the usual cause, in which case deleting
  the token file forces a fresh one on the next run.

Downloads stop at the first failed record, and no zipfile is written.
Requests time out after ``ESMDownloader.TIMEOUT`` seconds (120 by
default); raise it on a slow connection or when requesting long records.

Be considerate with the service: one request is issued per record, so a
40-record suite means 40 requests. Download a suite once and keep the
zipfile rather than re-fetching it on every run.
