SLF: PFA inventory with correlations
=====================================

This example generates storey loss functions from a PFA-based component
inventory with a separate correlation file and no grouping
(Weibull regression, seed 1).

Corresponds to test case: ``('pfa_inv', 'pfa_corr', False, False, "weibull", 1)``.

Input files
-----------

Download the input files from the repository:

* `pfa_inv.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/slf/assets/pfa_inv.json>`_
  — component inventory (PFA-driven, non-structural components).
* `pfa_corr.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/slf/assets/pfa_corr.json>`_
  — damage-state correlation matrix for the PFA components.

The inventory structure is identical to the PSD case. Components here
use ``"EDP": "PFA"`` and median demands expressed in units of g
(peak floor acceleration).

Running the example
-------------------

.. code-block:: python

    import json
    from djura.slf import SLF
    from djura.slf.utilities import filter_args

    with open("pfa_inv.json", encoding="utf-8") as f:
        inventory = json.load(f)

    with open("pfa_corr.json", encoding="utf-8") as f:
        correlations = json.load(f)

    data = {
        "inventory": inventory,
        "correlations": correlations,
        "do_grouping": False,
        "include_correlations": False,
        "conversion": "1.0",
        "realizations": "20",
        "replacement_cost": "1.0",
        "regression": "weibull",
        "storey": None,
        "directionality": None,
        "seed": "1",
    }

    # Correlations are only used when include_correlations=True.
    data["correlations"] = None

    outs = {}
    n_prev = 0

    unique_edp = set(item["EDP"] for item in inventory)
    for edp in unique_edp:
        data["edp"] = edp.lower()
        data["inventory"] = [item for item in inventory if item["EDP"] == edp]

        id_map = {old: new for new, old in enumerate(
            sorted(item["id"] for item in data["inventory"]))}
        for item in data["inventory"]:
            item["id"] = id_map[item["id"]]

        filtered_data = filter_args(SLF, data)
        filtered_data["n_prev"] = n_prev

        slf_obj = SLF(**filtered_data)
        out = slf_obj.generate_slfs()

        outs.update(out)
        n_prev += len(data["inventory"])
        del slf_obj

    print(outs)
