SLF: PSD inventory with correlations and grouping
==================================================

This example generates storey loss functions from a PSD-based component
inventory with damage-state correlations enabled and component grouping
active (Weibull regression, seed 0).

Corresponds to test case: ``('inventory', 'correlations', False, True, "weibull", 0)``.

Input files
-----------

Download the input files from the repository:

* `inventory.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/slf/assets/inventory.json>`_
  — component inventory (PSD-driven, structural and non-structural components,
  multiple damage states per component).
* `correlations.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/slf/assets/correlations.json>`_
  — damage-state correlation matrix between components.

Each entry in ``inventory.json`` defines one component with the following fields:

* ``EDP``: engineering demand parameter (``"PSD"`` or ``"PFA"``)
* ``Component``: component class (``"S"`` structural, ``"NS"`` non-structural)
* ``Group``: grouping label used when ``do_grouping=True``
* ``Quantity``: number of units per storey
* ``damage-states``: number of damage states
* ``median-demand``: median EDP at each damage state threshold
* ``total-dispersion``: total lognormal dispersion at each threshold
* ``repair-cost``: unit repair cost per damage state
* ``cost-dispersion``: dispersion on repair cost

Running the example
-------------------

.. code-block:: python

    import json
    from djura.slf import SLF
    from djura.slf.utilities import filter_args

    with open("inventory.json", encoding="utf-8") as f:
        inventory = json.load(f)

    with open("correlations.json", encoding="utf-8") as f:
        correlations = json.load(f)

    data = {
        "inventory": inventory,
        "correlations": correlations,
        "do_grouping": True,
        "include_correlations": False,
        "conversion": "1.0",
        "realizations": "20",
        "replacement_cost": "1.0",
        "regression": "weibull",
        "storey": None,
        "directionality": None,
        "seed": "0",
    }

    # Correlations are only used when include_correlations=True.
    # Set to None otherwise so SLF does not apply them.
    data["correlations"] = None

    outs = {}
    n_prev = 0

    unique_edp = set(item["EDP"] for item in inventory)
    for edp in unique_edp:
        data["edp"] = edp.lower()
        data["inventory"] = [item for item in inventory if item["EDP"] == edp]

        # Re-index component IDs to be contiguous from 0
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
