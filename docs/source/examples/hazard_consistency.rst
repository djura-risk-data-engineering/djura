Hazard consistency checking
===========================

This example verifies that a set of selected ground motion records is
consistent with the site hazard curve across multiple intensity measures
(IMs) and hazard levels. It uses OQ Engine hazard curve output as input.

Overview
--------

The workflow has three steps:

1. Parse the OQ hazard curve files into a structured dictionary with
   ``proc_oq_hazard_curve``.
2. For each hazard level (PoE), load the IM values of the selected
   records with ``get_rs_imi_intensities``.
3. Run ``HazardConsistency.check`` to compare the empirical record
   distribution against the target hazard curve.

Input data
----------

You need:

* **OQ hazard curve files** and **hazard.json** in a directory.
  Download the full folder:
  `tests/hazard/assets/hazard/ <https://github.com/djura-risk-data-engineering/djura/tree/main/tests/hazard/assets/hazard>`_
* **Selected record IM files**: one JSON per hazard level
  (``records_<poe>.json``), organised in one subdirectory per
  conditioning IM. Download the full folder:
  `tests/hazard/assets/records/ <https://github.com/djura-risk-data-engineering/djura/tree/main/tests/hazard/assets/records>`_

Expected directory layout::

    hazard/
        hazard.json
        hazard_curve-mean-PGA_4.csv
        hazard_curve-mean-SA(1.5)_4.csv
        ...
    records/
        SA(1.5)/
            records_0.4.json
            records_0.2.json
            records_0.1.json
            ...

Running the check
-----------------

.. code-block:: python

    from pathlib import Path
    from djura.record_selection.utilities import (
        proc_oq_hazard_curve,
        get_rs_imi_intensities,
    )
    from djura.hazard_consistency import HazardConsistency

    data_path = Path("path/to/assets")

    # PoEs at which the hazard curves are evaluated (annual rates).
    # These match the return periods used in the OQ PSHA run.
    poes = [
        0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
        0.0025, 0.001, 0.0004, 0.0002, 0.0001,
    ]

    # Step 1: parse OQ hazard curve output.
    # proc_oq_hazard_curve reads all hazard_curve-mean-*.csv files in
    # the directory and returns conditional IM levels and PoEs per IMT.
    hz = proc_oq_hazard_curve(
        poes,
        data_path / "hazard",
        json_file=data_path / "hazard/hazard.json",
    )

    # IMs to check consistency for (must match the OQ hazard output).
    imts_to_check = ["PGA", "FIV3(1.0)", "Sa_avg2(0.5)", "SA(1.5)", "PGV"]

    # Conditioning IM used during record selection.
    conditioning_imt = "SA(1.5)"

    # Step 2 + 3: build HazardConsistency model and check each IM.
    cond_imls = hz["cond_imls"][conditioning_imt]
    cond_poes = hz["cond_poes"]
    inv_t     = hz["investigation_time"]

    model = HazardConsistency(
        conditional_intensities=cond_imls,
        conditional_poes=cond_poes,
        investigation_time=inv_t,
    )

    results = {}
    for imi in imts_to_check:
        # Load the IM values of selected records at each hazard level.
        imls = get_rs_imi_intensities(
            selection_dir=data_path / f"records/{conditioning_imt}",
            poes=cond_poes,
            imi=imi,
        )

        # check() compares the empirical record distribution to the
        # target hazard curve and returns goodness-of-fit statistics.
        results[imi] = model.check(imls, num_im=500)

    print(results)

Output
------

``results`` is a dictionary keyed by IM name. Each entry contains the
hazard consistency metrics for that IM across all checked hazard levels,
which can be used to assess whether the selected suite adequately
represents the site hazard.

Supported IMs
-------------

The IMs passed to ``get_rs_imi_intensities`` must match those available
in the selected record files. Typical values used with djura are:

* Spectral acceleration: ``SA(T)``
* Average spectral acceleration: ``Sa_avg(T)``, ``Sa_avg2(T)``
* Peak ground acceleration: ``PGA``
* Peak ground velocity: ``PGV``
* Filtered incremental velocity: ``FIV3(T)``
