EDP-IM: masonry infill frame
=============================

Predicts EDP-IM curves for a masonry infill frame using
``edp_im_infill``. The pushover curve is loaded from two text files
(lateral forces and storey displacements) and the backbone is built
via ``get_infill_spo``.

Corresponds to ``test_infill`` with ``im_type="sa"``.

Input files
-----------

* `infill.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/infill.json>`_
  — damping, modal mass (``mstar``), and participation factor (``gamma``).
* `spo_disps_infill.txt <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/spo/spo_disps_infill.txt>`_
  — storey displacement matrix from pushover (rows = load steps).
* `spo_shear_infill.txt <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/spo/spo_shear_infill.txt>`_
  — base shear vector from pushover.

Running the example
-------------------

.. code-block:: python

    import json
    import numpy as np
    from djura.vulnerability_modeller.backbone import Backbone, get_infill_spo
    from djura.vulnerability_modeller.utilities import to_json_serializable
    from djura.edp_im.predict import edp_im_infill

    with open("infill.json", encoding="utf-8") as f:
        data = json.load(f)

    disp = np.genfromtxt("spo_disps_infill.txt")
    rx   = np.genfromtxt("spo_shear_infill.txt")

    spo = {"force": rx, "displacement": disp}

    body = {
        "backbone": {
            "spo": to_json_serializable(spo),
            "damping": data["damping"],
            "gamma": data["gamma"],
            "mstar": data["mstar"],
            "sdof": False,
        },
        "im_type": "sa",
        "backboneMethod": "spo",
    }

    # Build the infill backbone and attach the SPO curve
    bb = dict(Backbone(body["backbone"], "spo", "infill").backbone)
    bb["im_type"] = body["im_type"]
    bb["r-plot"], bb["mu-plot"] = get_infill_spo(bb)

    result = edp_im_infill(bb)

    print("Status:", result["status"])

SDOF normalisation
------------------

If the pushover forces and displacements are already normalised to SDOF
quantities, set ``"sdof": True`` and apply the appropriate factors
before passing to ``spo``:

.. code-block:: python

    f_factor = 800 * 1.3
    d_factor = 1.3

    spo = {
        "force": rx / f_factor,
        "displacement": disp / d_factor,
    }

    body["backbone"]["spo"] = to_json_serializable(spo)
    body["backbone"]["sdof"] = True
