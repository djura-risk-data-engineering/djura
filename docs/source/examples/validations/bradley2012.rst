Validation: GCIM-based selection algorithm (Bradley, 2012)
==========================================================

This page documents the replication of the application that introduced the
GCIM-based selection algorithm. It is intended both as evidence that the
selection behaves as its literature describes, and as a worked example of how
the weight vector shapes a suite.

    Bradley, B. A. (2012). A ground motion selection algorithm based on the
    generalized conditional intensity measure approach. *Soil Dynamics and
    Earthquake Engineering*, 40, 48-61. DOI:
    `10.1016/j.soildyn.2012.04.007 <https://doi.org/10.1016/j.soildyn.2012.04.007>`_

The case lives in
`tests/rs/validations/test_case2_bradley2012.py <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/validations/test_case2_bradley2012.py>`_.


The case
--------

A site in Los Angeles at -118.243 and 34.053, with :math:`V_{s30}` = 760 m/s
and :math:`Z_{2.5}` = 1.0 km, conditioned on :math:`Sa(3.0\,\mathrm{s})` at
three exceedance levels. Thirty records are selected per suite.

The result of the article is the effect of the weight vector. Weighting
spectral acceleration alone leaves the CAV and duration distributions of the
selected suite biased, and moving thirty per cent of the weight onto CAV and
the two significant durations removes the bias while matching the spectral
ordinates just as well.

.. list-table::
   :header-rows: 1
   :widths: 12 20 26 42

   * - Sub-case
     - Exceedance
     - Weights
     - Purpose
   * - ``2A``
     - 10 % in 50 yr
     - equation (15), SA only
     - figure 2, the biased CAV and duration distributions
   * - ``2B``
     - 10 % in 50 yr
     - equation (16)
     - figure 3, unbiased across all measures
   * - ``2C``
     - 50 % in 50 yr
     - equation (16)
     - figures 5 and 6, tables 3 and 5
   * - ``2D``
     - 1 % in 50 yr
     - equation (16)
     - figures 5 and 6, tables 3 and 5

The intensity measure vector runs to sixteen entries: spectral accelerations at
0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0 and 5.0 s, together with PGA, PGV, ASI, SI,
DSI, CAV, :math:`Ds_{575}` and :math:`Ds_{595}`.
:math:`Sa(3.0\,\mathrm{s})` is the conditioning measure and is not an entry.


Conditioning amplitudes
-----------------------

The published targets give the conditioning amplitude of each level exactly.
These are used rather than the rounded values of the article's text:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Exceedance
     - :math:`Sa(3.0\,\mathrm{s})` [g]
     - As printed in the article
   * - 50 % in 50 yr
     - 0.03749344
     - 0.038
   * - 10 % in 50 yr
     - 0.08270873
     - 0.083
   * - 1 % in 50 yr
     - 0.16783433
     - 0.17


Seismic hazard input
--------------------

The article states that its targets were built from the full disaggregated
rupture set rather than from a single mean scenario, the algorithm drawing a
random rupture from :math:`P(Rup \mid IM_j)` before drawing a realization from
it. The author confirms that a rupture-by-rupture listing was never archived,
those probabilities having been computed internally by the hazard software.

The asset ``bradley2012_deagg_sa3pt0.json`` therefore stands in for it,
recovered from the bars of the published deaggregation: 62 weighted
magnitude-distance cells per level, taken at bin centres of 10 km and 0.5
magnitude units, with the epsilon bins summed. The parse is checked against the
documented means of the deaggregation:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 19 19

   * - Exceedance
     - Documented :math:`\bar{M}`
     - Recovered
     - Documented :math:`\bar{R}` [km]
     - Recovered
   * - 50 % in 50 yr
     - 6.993
     - 7.042
     - 44.51
     - 41.61
   * - 10 % in 50 yr
     - 7.073
     - 7.137
     - 35.16
     - 34.67
   * - 1 % in 50 yr
     - 7.103
     - 7.166
     - 30.95
     - 31.23

The recovered mean magnitude sits about 0.06 units high at every level, which
is what collapsing a 0.5-wide bin onto its centre produces.

The OpenSHA configuration behind the deaggregation is recorded with the asset:
Boore and Atkinson (2008), GMRotI50, total sigma, no truncation; the USGS/CGS
1996 Adjusted California earthquake rupture forecast with Frankel's fault model
and background seismicity included; maximum distance 200 km.


Why the target is not reproduced exactly
----------------------------------------

Working from the binned deaggregation rather than the internal rupture set has
two consequences. Both are bounded, both are identified by their signature, and
neither is a defect in djura.

**Truncation.** The plotted deaggregation stops at 110 km, so the recovered
bars carry 96.6, 99.2 and 99.8 per cent of the hazard at the three levels. The
missing part is the distant tail, and dropping it raises the medians and
narrows the distribution. The error tracks the truncation, which is how it is
identified:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - 50 % in 50 yr
     - 10 % in 50 yr
     - 1 % in 50 yr
   * - Hazard captured by the bars
     - 96.6 %
     - 99.2 %
     - 99.8 %
   * - PGV dispersion, djura / article
     - 0.55
     - 0.86
     - 0.92
   * - CAV dispersion, djura / article
     - 0.61
     - 0.89
     - 0.96
   * - Worst median, djura / article
     - 1.31
     - 1.10
     - 0.92

**Binning.** Collapsing each cell onto its centre removes the spread within it,
which accounts for a residual five to seven per cent dispersion deficit at the
1 per cent level, where truncation is negligible. It touches only the
between-rupture term of the variance, so the measures most strongly correlated
with the conditioning one are untouched: DSI, :math:`Sa(2.0\,\mathrm{s})` and
:math:`Sa(5.0\,\mathrm{s})` all reproduce their dispersion to within half a per
cent at every level.

Accordingly the medians are asserted, at 15 % for the two rarer levels and 35 %
for the 50 per cent level whose rupture set is the most truncated; the
dispersions are reported rather than asserted.


Ground motion and correlation models
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - IM
     - Model in the article
     - djura class
   * - :math:`SA`, PGA, PGV
     - Boore and Atkinson (2008)
     - ``BooreAtkinson2008``
   * - :math:`ASI`
     - Bradley (2010)
     - ``Bradley2010ASI``
   * - :math:`SI`
     - Bradley et al. (2009)
     - ``BradleyEtAl2009SI``
   * - :math:`DSI`
     - Bradley (2011)
     - ``Bradley2011DSI``
   * - :math:`CAV`
     - Campbell and Bozorgnia (2010)
     - ``CampbellBozorgnia2008``
   * - :math:`Ds_{575}`, :math:`Ds_{595}`
     - Bommer et al. (2009)
     - ``BommerEtAl2009RSD``

The CAV row is not a substitution. djura carries the Campbell and Bozorgnia
(2010) CAV equation, which the article cites, inside its
``CampbellBozorgnia2008`` implementation; the two share a functional form,
which is why the article notes that the CAV-PGA correlation is consistent
between them.

The article's correlation models are named explicitly, three of these pairs
having more than one model registered:

.. code-block:: json

   "correlation-models": {
     "SA-SA": "baker_jayaram",
     "Ds575-SA": "bradley2011_ds575_sa",
     "Ds595-SA": "bradley2011_ds595_sa"
   }

With those supplied, table 4 of the article is reproduced exactly:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - IM
     - Published :math:`\rho`
     - djura
     - Difference
   * - :math:`Sa(0.1)`
     - 0.066
     - 0.0664
     - +0.0004
   * - :math:`Sa(0.3)`
     - 0.254
     - 0.2535
     - -0.0005
   * - :math:`Sa(1.0)`
     - 0.609
     - 0.6087
     - -0.0003
   * - :math:`PGV`
     - 0.758
     - 0.7578
     - -0.0002
   * - :math:`CAV`
     - 0.525
     - 0.5246
     - -0.0004
   * - :math:`Ds_{595}`
     - 0.111
     - 0.1109
     - -0.0001


Weight vectors
--------------

The two vectors of the article, and the contrast between them, are the point of
the case.

.. math::

   \text{equation (15):}\quad & w_i = \tfrac{1}{9}
       \ \text{on each } SA \text{ ordinate},\ 0 \text{ elsewhere} \\
   \text{equation (16):}\quad & w_i = 0.7 \times \tfrac{1}{9}
       \ \text{on each } SA \text{ ordinate},\
       0.1 \text{ on each of } CAV,\ Ds_{575},\ Ds_{595}

:math:`Sa(10\,\mathrm{s})` is dropped from the article's vector of seventeen,
so the weights are renormalised over the eight ordinates retained. This
preserves the seventy-thirty split between the spectral and the other measures,
which is the quantity the article studies.

``im_weights`` is a flat list aligned element by element with the expanded
``imi``.


Input file
----------

The complete configuration is
``tests/rs/validations/assets/bradley2012_input.json``, the 10 per cent level
with the weights of equation (16) and the 62 ruptures inline. Its settings, the
rupture list aside:

.. code-block:: json

   {
     "gmms": [
       {"SA": {"names": ["BooreAtkinson2008"], "weights": [1]}},
       {"PGA": {"names": ["BooreAtkinson2008"], "weights": [1]}},
       {"PGV": {"names": ["BooreAtkinson2008"], "weights": [1]}},
       {"ASI": {"names": ["Bradley2010ASI"], "weights": [1],
                "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
       {"SI": {"names": ["BradleyEtAl2009SI"], "weights": [1],
               "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
       {"DSI": {"names": ["Bradley2011DSI"], "weights": [1],
                "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
       {"CAV": {"names": ["CampbellBozorgnia2008"], "weights": [1]}},
       {"Ds575": {"names": ["BommerEtAl2009RSD"], "weights": [1]}},
       {"Ds595": {"names": ["BommerEtAl2009RSD"], "weights": [1]}}
     ],
     "site-parameters": {
       "vs30": 760, "z2pt5": 1.0, "ztor": 3, "dip": 45,
       "mechanism": "reverse fault"
     },
     "im-star": {
       "type": "SA(3.0)",
       "value": 0.08270873,
       "gmms": {"names": ["BooreAtkinson2008"], "weights": [1]}
     },
     "num-components": 2,
     "component-definition": "geomean",
     "num_records": 30,
     "nreplicate": 1,
     "seed": 1,
     "ks_alpha": 0.1,
     "max_scaling_factor": 10,
     "context_limits": {}
   }

Three settings are worth explaining.

``context_limits: {}``
   No causal parameter screening, which is explicit in the article and central
   to its argument: GCIM recovers sensible magnitudes and distances without any
   screen. Adding one invalidates the case.

``max_scaling_factor: 10``
   Must not bind. The SA-only suite of the article has a median factor of 1.9
   with a 90th percentile of 9.0, so a limit of 3 would silently destroy the
   comparison. The test asserts that the largest factor used stays below the
   limit.

``ztor``, ``dip`` and ``mechanism``
   None is stated by the article. Reverse faulting is inferred from the site and
   the near-source contribution, and ``ztor`` is assumed at 3 km. ``dip`` is
   required by ``CampbellBozorgnia2008`` but is inert here: with the
   Joyner-Boore and rupture distances equal and ``ztor`` at least 1, the
   hanging-wall distance term is zero, so the whole hanging-wall term, the only
   place ``dip`` enters, vanishes.


What is validated
-----------------

**The target.** The published median of all sixteen measures at all three
levels, within the tolerances set out above.

**The result of the article.** Against djura's own target median, the suite
selected on spectral acceleration alone overshoots CAV by 1.6,
:math:`Ds_{575}` by 2.6 and :math:`Ds_{595}` by 2.7, and the fuller vector
brings all three back to within nine per cent. Both suites match the spectral
ordinates to within twelve per cent, which is what makes the comparison a fair
one.

.. list-table:: Median of the suite over the target median, 10 % in 50 yr
   :header-rows: 1
   :widths: 34 33 33

   * - IM
     - equation (15)
     - equation (16)
   * - :math:`CAV`
     - 1.63
     - 0.99
   * - :math:`Ds_{575}`
     - 2.55
     - 0.88
   * - :math:`Ds_{595}`
     - 2.73
     - 1.01
   * - :math:`Sa(0.2\,\mathrm{s})`
     - 0.97
     - 1.12
   * - :math:`Sa(2.0\,\mathrm{s})`
     - 1.02
     - 1.11

The Kolmogorov-Smirnov outcomes follow: the SA-only suite is rejected for
:math:`ASI`, :math:`CAV`, :math:`Ds_{575}` and :math:`Ds_{595}`, four of
sixteen, and the fuller vector for :math:`CAV` alone.

**The second finding of the article.** The fuller vector needs less amplitude
scaling, a median factor of 1.2 against 3.5, where the article reports 1.1
against 1.9. The ordering is asserted; the levels depend on the database.


Running the case
----------------

.. code-block:: bash

   # assertions only
   pytest tests/rs/validations/test_case2_bradley2012.py -m validation

   # tabulate the computed values against the published ones
   pytest tests/rs/validations/test_case2_bradley2012.py -m validation --report


Limitations
-----------

* **The rupture sets are recovered, not published.** The internal
  rupture-by-rupture probabilities were never archived, so the binned
  deaggregation stands in for them, with the consequences set out above. The
  50 per cent level is the weakest of the three.
* :math:`Sa(10\,\mathrm{s})` **is dropped**, so one published column is not
  checked. The correlations of the period-independent measures with SA guard
  with ``0.01 <= period < 10``, excluding the endpoint.
* **Duration runs six to seventeen per cent low** at every level, including
  where truncation is negligible. ``BommerEtAl2009RSD`` reads ``ztor``, which
  the article does not state and which is assumed here.
* **The causal parameters of the suite are not comparable.** The bundled
  database holds few large-magnitude records where the article used NGA-West1,
  so the algorithm reaches for smaller ones and scales harder, giving mean
  magnitudes 0.2 to 0.4 units below the published suites and wider distance
  dispersions. What survives the change of database, and is asserted, is the
  contrast between the two vectors.
* **Table 3 is not compared record by record.** The median intensity measures
  of the published suites depend on its record set, so the comparison is
  against the target instead.
