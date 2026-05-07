EDP-IM: base-isolated structure
================================

Predicts EDP-IM curves for a base-isolated structure using
``edp_im_isol``. The backbone is defined by the isolation system
strength ratio ``R`` and post-yield displacement ``mu``.

Corresponds to ``test_isol``.

Input file
----------

* `isol.json <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/vm/assets/backbones/isol.json>`_
  — isolation system parameters (``R``: strength ratio, ``mu``:
  post-yield displacement in metres).

Running the example
-------------------

.. code-block:: python

    import json
    from djura.edp_im.predict import edp_im_isol

    with open("isol.json", encoding="utf-8") as f:
        data = json.load(f)

    # data = {"backbone": {"R": 4, "mu": 0.03}}
    result = edp_im_isol(data["backbone"])

    print("Status:", result["status"])

Or inline, without a file:

.. code-block:: python

    from djura.edp_im.predict import edp_im_isol

    result = edp_im_isol({"R": 4, "mu": 0.03})

    print("Status:", result["status"])
