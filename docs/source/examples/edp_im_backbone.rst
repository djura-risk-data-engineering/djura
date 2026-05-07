EDP-IM: bilinear system with file-based backbone
=================================================

Predicts EDP-IM curves for a bilinear system where the backbone is
provided as a JSON file rather than defined inline. Three backbone input
methods are supported: ``"backbone"`` (parametric), ``"idealized"``
(force-displacement points), and ``"spo"`` (pushover curve).

Corresponds to ``test_backbone`` with ``backbone1.json``.

Input files
-----------

* `backbone1.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/backbone1.json>`_
  — parametric backbone (damping, ductility, hardening ratio, period).
* `backbone2.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/backbone2.json>`_,
  `backbone3.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/backbone3.json>`_
  — idealized force-displacement backbone.
* `backbone4.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/backbone4.json>`_,
  `backbone5.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/backbone5.json>`_
  — SPO pushover backbone (requires separate SPO text files).

Parametric backbone (``"backbone"`` method)
-------------------------------------------

.. code-block:: python

    import json
    from djura.vulnerability_modeller.backbone import Backbone
    from djura.edp_im.predict import edp_im

    with open("backbone1.json", encoding="utf-8") as f:
        data = json.load(f)

    backbone = Backbone(data, "backbone", "bilin").backbone

    result = edp_im({
        "method": "shahnazaryan-oreilly",
        "hysteresis": "bilin",
        "im_type": "sa",
        "backboneMethod": "backbone",
        "backbone": backbone,
    })

    print("Status:", result["status"])

SPO pushover backbone (``"spo"`` method)
----------------------------------------

When the backbone is derived from a pushover curve, load the force and
displacement arrays from text files and attach them to the backbone data:

.. code-block:: python

    import json
    import numpy as np
    from djura.vulnerability_modeller.backbone import Backbone
    from djura.vulnerability_modeller.utilities import to_json_serializable
    from djura.edp_im.predict import edp_im

    with open("backbone4.json", encoding="utf-8") as f:
        data = json.load(f)

    # spo_topdisp.txt: columns are [force, disp_storey1, disp_storey2, ...]
    spo = np.loadtxt("spo_topdisp.txt")
    data["spo"] = {
        "force": spo[:, 0],
        "displacement": spo[:, 1:],
    }

    backbone = Backbone(
        to_json_serializable(data), "spo", "bilin"
    ).backbone

    result = edp_im({
        "method": "shahnazaryan-oreilly",
        "hysteresis": "bilin",
        "im_type": "sa",
        "backboneMethod": "spo",
        "backbone": backbone,
    })

    print("Status:", result["status"])

Using average spectral acceleration (Sa_avg)
--------------------------------------------

Replace ``"im_type": "sa"`` with ``"im_type": "sa_avg"`` in any of the
calls above to predict EDP-IM relationships in terms of average spectral
acceleration instead of Sa(T):

.. code-block:: python

    result = edp_im({
        "method": "shahnazaryan-oreilly",
        "hysteresis": "bilin",
        "im_type": "sa_avg",
        "backboneMethod": "backbone",
        "backbone": backbone,
    })
