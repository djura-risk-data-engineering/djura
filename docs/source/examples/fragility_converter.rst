Fragility and vulnerability model conversion
=============================================

This example shows how to convert a seismic fragility model from one
intensity measure (IM) to another using ``FF`` (exact conversion via OQ
disaggregation context) and ``FFApproximate`` (closed-form approximation).

Both classes are in ``djura.fragility_converter``.

Exact conversion with ``FF``
----------------------------

``FF`` converts a lognormal fragility function defined in IM1 to an
equivalent one in IM2, using the joint IM distribution derived from an
OQ PSHA disaggregation context. The fragility parameters (median and
dispersion) are read from a JSON file and the conversion is run for a
discrete grid of IM2 values.

**Case 1: SA(1.0) to SA(0.5)**

.. code-block:: python

    import json
    import pickle
    from djura.fragility_converter import FF

    # Base input: GMM ensemble and site/rupture parameters
    with open("input.json", encoding="utf-8") as f:
        base_input = json.load(f)

    # OQ disaggregation context (same format as the record selection example)
    with open("ctx.pickle", "rb") as f:
        oq_data = pickle.load(f)

    # Source fragility: median and dispersion in SA(1.0)
    im1 = {
        "median": 0.376,
        "dispersion": 0.295,
        "name": "SA(1.0)",
    }

    # Target IM: SA(0.5), evaluated over a grid from 0.001 to 5.0 g
    im2 = {
        "name": "SA(0.5)",
        "min": 0.001,
        "max": 5.0,
        "num_pts": 100,
    }

    ff = FF(im1, im2, data=base_input, dis_oq=oq_data)
    im2_probs, im2_range = ff.create()

``im2_probs`` contains the exceedance probabilities on the ``im2_range``
grid, representing the converted fragility curve in SA(0.5).

**Case 2: SA(0.5) to SA(1.0)**

The conversion is symmetric — swap ``im1`` and ``im2``:

.. code-block:: python

    im1 = {
        "median": 0.194,
        "dispersion": 0.295,
        "name": "SA(0.5)",
    }

    im2 = {
        "name": "SA(1.0)",
        "min": 0.001,
        "max": 5.0,
        "num_pts": 100,
    }

    ff = FF(im1, im2, data=base_input, dis_oq=oq_data)
    im2_probs, im2_range = ff.create()

Approximate conversion with ``FFApproximate``
---------------------------------------------

``FFApproximate`` uses a closed-form approximation that does not require
a full disaggregation context. Instead it takes a pre-computed input
file that already encodes the IM correlation structure for the site.

.. code-block:: python

    import json
    from djura.fragility_converter import FFApproximate

    # Pre-computed approximation input (encodes IM correlations for the site)
    with open("approx_SA(0.5).json", encoding="utf-8") as f:
        approx_input = json.load(f)

    # Pre-existing fragility functions in multiple IMs
    with open("ff.json", encoding="utf-8") as f:
        ff_data = json.load(f)

    imt1, imt2 = "SA(1.0)", "SA(0.5)"
    ds = "1"                           # damage state key
    frag1 = ff_data[imt1][ds]

    im1 = {
        "median": frag1["median"],
        "dispersion": frag1["beta"],
        "name": imt1,
    }

    im2 = {"name": imt2}

    ff = FFApproximate(im1, im2, data=approx_input)
    im2_probs, im2_range = ff.create()

Accuracy metrics
----------------

After conversion, the output ``im2_probs`` and ``im2_range`` can be
compared against a reference fragility curve by fitting a lognormal CDF
to the converted probabilities and computing the Hellinger distance
between the fitted and reference parameters.

.. code-block:: python

    import numpy as np
    from djura.record_selection.utilities import fit_cdf_to_data
    from djura.record_selection.metrics import hellinger_distance

    # Fit a lognormal CDF to the converted fragility curve.
    # method can be "least_squares" or "mle"
    fit = fit_cdf_to_data(
        im2_range, im2_probs,
        distribution="lognormal",
        method="least_squares",
    )

    mu_fit = fit["mu"]      # log-space mean (ln median)
    s_fit  = fit["sigma"]   # log-space standard deviation (dispersion)

    # Reference fragility parameters in IM2 (if known)
    ref_median     = 0.194   # median in IM2 units
    ref_dispersion = 0.295

    h = hellinger_distance(ref_median, ref_dispersion,
                           np.exp(mu_fit), s_fit)

    print(f"Fitted median     : {np.exp(mu_fit):.3f}")
    print(f"Fitted dispersion : {s_fit:.3f}")
    print(f"Hellinger distance: {np.round(h[0], 3)}")

The Hellinger distance ranges from 0 (identical distributions) to 1
(completely disjoint). Values below 0.05 indicate a very close match
between the converted and reference fragility curves.

You can also use ``"mle"`` instead of ``"least_squares"`` as the
fitting method and compare which produces a lower Hellinger distance
for your specific case.

When to use each approach
--------------------------

* **FF**: use when you have an OQ PSHA disaggregation context available.
  Produces the most accurate conversion because it uses the full joint
  IM distribution from the site hazard model.
* **FFApproximate**: use when no disaggregation context is available or
  when a fast closed-form estimate is sufficient. Requires a
  pre-computed approximation input file for the site.
