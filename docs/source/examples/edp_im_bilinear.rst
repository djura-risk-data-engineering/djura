EDP-IM: bilinear system with analytical backbone
=================================================

Predicts strength-ratio and ductility demand curves for a bilinear SDOF
using the Shahnazaryan-O'Reilly method. Backbone parameters are defined
directly as key-value pairs — no external file needed.

Corresponds to ``test_batch`` with ``method="shahnazaryan-oreilly"``.

Running the example
-------------------

.. code-block:: python

    from djura.vulnerability_modeller.backbone import Backbone
    from djura.edp_im.predict import edp_im, edp_im_batch

    # Single building
    backbone_params = {
        "damping": 0.075,
        "ductility": 3,
        "hardening_ratio": 0.05,
        "period": 1.0,
        "ductility_f": 8,
    }

    backbone = Backbone(backbone_params, "backbone", "bilin").backbone

    result = edp_im({
        "method": "shahnazaryan-oreilly",
        "hysteresis": "bilin",
        "im_type": "sa",
        "backboneMethod": "backbone",
        "backbone": backbone,
    })

    print("Status:", result["status"])

Multiple buildings can be processed in one call with ``edp_im_batch``:

.. code-block:: python

    result = edp_im_batch([
        {
            "hysteresis": "bilin",
            "im_type": "sa",
            "method": "shahnazaryan-oreilly",
            "backboneMethod": "backbone",
            "backbone": {
                "damping": 0.02,
                "ductility": 3,
                "ductility_f": 6,
                "hardening_ratio": 0.05,
                "period": 1.0,
            },
        },
        {
            "hysteresis": "bilin",
            "im_type": "sa",
            "method": "shahnazaryan-oreilly",
            "backboneMethod": "backbone",
            "backbone": {
                "damping": 0.03,
                "ductility": 3,
                "ductility_f": 6,
                "hardening_ratio": 0.02,
                "period": 0.5,
            },
        },
    ])

    print("Success IDs:", result["success_ids"])
    print("Failure IDs:", result["failure_ids"])

Supported methods
-----------------

The ``method`` key selects the R-mu-T relationship used to convert
spectral acceleration to ductility demand. Available options:

* ``"shahnazaryan-oreilly"`` — ML-based, recommended for general use
* ``"ec8"`` — Eurocode 8
* ``"krawinkler_nassar"``
* ``"miranda"``
* ``"newmark_hall"``
* ``"vidic"``
* ``"guerrini"``
