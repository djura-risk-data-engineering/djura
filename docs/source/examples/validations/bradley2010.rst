Validation: the GCIM approach (Bradley, 2010)
=============================================

This page documents the replication of the worked example that introduced the
generalized conditional intensity measure approach. It is intended both as
evidence that the machinery behaves as its literature describes, and as a
worked example of a selection whose intensity measure vector is not confined to
spectral accelerations.

    Bradley, B. A. (2010). A generalized conditional intensity measure approach
    and holistic ground-motion selection. *Earthquake Engineering & Structural
    Dynamics*, 39(12), 1321-1342. DOI:
    `10.1002/eqe.995 <https://doi.org/10.1002/eqe.995>`_

The case lives in
`tests/rs/validations/test_case1_bradley2010.py <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/validations/test_case1_bradley2010.py>`_.


The case
--------

A rock site in Christchurch, New Zealand, at 172.6E and 43.5S, with
:math:`V_{s30}` = 760 m/s, conditioned on :math:`Sa(1.0\,\mathrm{s})` =
0.165 g, the 2 % in 50 year value. The intensity measure vector is
:math:`\{SA, SI, ASI, I_a, Ds_{595}\}`, which is what makes the case worth
replicating: it exercises the period-independent measures and the correlations
between them, not only the spectral ordinates.

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - Sub-case
     - Meaning
     - Status
   * - ``1A``
     - the target, figures 2 and 3
     - replicated
   * - ``1B``
     - suite 1, causal magnitude below 6.0 and distance below 20 km
     - replicated as a distribution, not as a record list
   * - ``1C``
     - suite 2, magnitude above 7.0 and distance above 50 km
     - replicated as a distribution, not as a record list
   * - ``1F``
     - focal mechanism, figure 4(b)
     - published value reproduced exactly; not reachable from the bundled
       database, see `Limitations`_


Seismic hazard input
--------------------

Figure 1 of the article is a magnitude-distance-epsilon deaggregation of
:math:`Sa(1.0\,\mathrm{s})` for the site, from the New Zealand national
seismic hazard model as implemented in OpenSHA. It is not tabulated in the
article, and the model is not bundled with djura, so the deaggregation was
obtained from the author: 1000 rupture scenarios with magnitude,
source-to-site distance, epsilon, rake and percent contribution, held as the
asset ``bradley2010_deagg_sa1p0_2pct50.csv``.

The percent contributions sum to 92.78, the list being truncated to the largest
contributors, and are normalised by that sum:

.. math::

   P(Rup \mid Sa(1.0) = 0.165\,\mathrm{g}) =
       \frac{\text{percent contribution}}{\sum \text{percent contribution}}

Three conventions of the original implementation are followed, all confirmed by
the author:

#. a single source-to-site distance per rupture, used as the distance input to
   every ground motion model, so ``rjb`` and ``rrup`` are both set to it;
#. no ``ztor`` and no ``d_hyp``, neither being an input to any model used here;
#. focal mechanism entering as a per-rupture ``rake``. A site-level
   ``mechanism`` is deliberately *not* set: djura derives rake from mechanism
   only when rake is absent, and the deaggregation spans three mechanisms.

Weighting the deaggregation gives a mean magnitude of 7.05, a mean distance of
63.5 km and a mean epsilon of 1.84, over ranges of :math:`M` 5.0 to 8.1 and
:math:`R` 10 to 191 km.


Ground motion and correlation models
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - IM
     - Model in the article
     - djura class
   * - :math:`SA`
     - Boore and Atkinson (2008)
     - ``BooreAtkinson2008``
   * - :math:`SI`
     - Bradley et al. (2009)
     - ``BradleyEtAl2009SI``
   * - :math:`ASI`
     - Bradley (2010)
     - ``Bradley2010ASI``
   * - :math:`I_a`
     - Travasarou et al. (2003)
     - ``TravasarouEtAl2003``
   * - :math:`Ds_{595}`
     - Abrahamson and Silva (1996)
     - ``AbrahamsonSilva1996``

Several of the correlation pairs needed here have more than one model
registered in djura, so the ones the article used are named explicitly:

.. code-block:: json

   "correlation-models": {
     "SA-SA": "baker_jayaram",
     "IA-SA": "baker2007_ia_sa",
     "Ds595-SA": "bradley2011_ds595_sa"
   }

With those supplied, table I of the article is reproduced:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - IM
     - Published :math:`\rho`
     - djura
     - Difference
   * - :math:`Sa(0.05)`
     - 0.42
     - 0.4157
     - -0.004
   * - :math:`Sa(0.5)`
     - 0.75
     - 0.7490
     - -0.001
   * - :math:`SI`
     - 0.92
     - 0.9160
     - -0.004
   * - :math:`ASI`
     - 0.61
     - 0.5866
     - -0.023
   * - :math:`I_a`
     - 0.70
     - 0.6970
     - -0.003

.. note::

   Omitting the ``correlation-models`` key changes the target materially. The
   registry lists a newer model first for ``SA-SA``, which gives
   :math:`\rho(Sa(0.05), Sa(1.0))` = 0.495 against the published 0.42, and for
   ``IA-SA``, which gives 0.600 against 0.70.

The :math:`ASI` difference is the one component model where djura implements a
later equation than the article, which used a pre-publication form later
published as Bradley (2011). It is not to be removed.

Two ways of taking epsilon
--------------------------

The article takes epsilon per rupture from the deaggregation itself, which the
New Zealand hazard model produced with its own ground motion model, while
predicting the intensity measures with Boore and Atkinson (2008). djura instead
derives epsilon from the model it is given for the conditioning measure, since
it carries neither that hazard model nor an input for supplying epsilon
directly. Either is a reasonable choice; they simply differ, and the difference
is confined to the dispersion above the conditioning period.

Both are therefore checked. Substituting the published epsilon into djura's own
per-rupture moments leaves only its ground motion model predictions and its
combination over the 1000 ruptures:

.. math::

   \bar{\mu} &= \sum_i p_i \left[
       \mu_i + \rho\,\sigma_i\,\varepsilon_i \right] \\
   \bar{\sigma}^2 &= \sum_i p_i \left[
       (1 - \rho^2)\sigma_i^2
       + \left( \mu_i + \rho\,\sigma_i\,\varepsilon_i
                - \bar{\mu} \right)^2 \right]

.. list-table:: Target with the published epsilon substituted
   :header-rows: 1
   :widths: 34 33 33

   * - Quantity
     - Median, djura / article
     - Dispersion, djura / article
   * - :math:`SA`, all 14 periods
     - 0.994 to 1.001
     - 1.000
   * - :math:`SI`
     - 1.004
     - 1.007
   * - :math:`ASI`
     - 0.984
     - 1.010
   * - :math:`I_a`
     - 1.012
     - 1.004
   * - :math:`Ds_{595}`, the article's own choices restored
     - 1.0000
     - 1.0000

Left to derive epsilon itself, djura matches the medians to within 1 % but
returns a smaller dispersion above the conditioning period, 0.378 against 0.628
at 1.5 s. The reason is structural: with self-consistent epsilon each rupture's
conditional mean passes through the conditioning amplitude as the correlation
approaches one, which removes the between-rupture variance that the epsilon of
the article retains. The deficit therefore tracks
:math:`(1 - \rho^2)`, deepest next to the conditioning period and recovering to
0.90 by 7.5 s. The medians are asserted; the dispersion is required only not to
*exceed* the published value, self-consistent epsilon being able to remove
variance but not add it.


Input file
----------

The complete configuration is
``tests/rs/validations/assets/bradley2010_input.json``, with the 1000 ruptures
inline. Its settings, the rupture list aside:

.. code-block:: json

   {
     "gmms": [
       {"SA": {"names": ["BooreAtkinson2008"], "weights": [1]}},
       {"SI": {"names": ["BradleyEtAl2009SI"], "weights": [1],
               "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
       {"ASI": {"names": ["Bradley2010ASI"], "weights": [1],
                "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
       {"IA": {"names": ["TravasarouEtAl2003"], "weights": [1]}},
       {"Ds595": {"names": ["AbrahamsonSilva1996"], "weights": [1]}}
     ],
     "correlation-models": {
       "SA-SA": "baker_jayaram",
       "IA-SA": "baker2007_ia_sa",
       "Ds595-SA": "bradley2011_ds595_sa"
     },
     "site-parameters": {"vs30": 760, "soil": 0},
     "im-star": {
       "type": "SA(1.0)",
       "value": 0.165,
       "gmms": {"names": ["BooreAtkinson2008"], "weights": [1]}
     },
     "num-components": 2,
     "component-definition": "geomean",
     "num_records": 15,
     "nreplicate": 1,
     "seed": 1,
     "ks_alpha": 0.1,
     "max_scaling_factor": 100,
     "context_limits": {}
   }

Four settings are worth explaining.

``soil: 0``
   The rock indicator required by ``AbrahamsonSilva1996``, which djura does not
   derive from ``vs30``. Without it :math:`Ds_{595}` cannot be predicted.

``num-components: 2`` with ``geomean``
   The article works with the geometric mean of the two horizontal components,
   and Boore and Atkinson predict GMRotI50. With one component djura stacks the
   two components as separate candidates and ignores the component definition
   entirely.

``ks_alpha: 0.1``
   Every Kolmogorov-Smirnov bound in figures 4 to 6 is drawn at the 10 %
   significance level, not djura's 5 % default.

``max_scaling_factor: 100``
   Effectively unbounded. Every record is scaled to the conditioning amplitude
   by construction, so the limit must not bind.

``imi`` carries spectral accelerations at fourteen periods together with
:math:`SI`, :math:`ASI`, :math:`I_a` and :math:`Ds_{595}`, so that one target
serves figures 2, 3 and 5. The article's five-measure vector is isolated
through ``im_weights``, which puts equal weight on
:math:`Sa(0.05)`, :math:`Sa(0.5)`, :math:`SI`, :math:`ASI`, :math:`I_a` and
:math:`Ds_{595}` and zero elsewhere. :math:`Sa(1.0)` is excluded from ``imi``
because the conditional distribution is degenerate at the conditioning period.


Selection
---------

Sub-cases 1B and 1C reproduce the causal bands of the two suites of table II.
The bands are the article's own screening, and are the one place in these cases
where ``context_limits`` is used deliberately: the argument of the article is
that two magnitude-distance screened suites misrepresent the GCIM target.

The selected records are not the article's, the algorithm drawing random
realizations and the database differing, so the comparison is of suite
statistics. On the bundled database the mean causal magnitude and distance fall
within 0.26 magnitude units and 19 km of the published suites, with no
screening beyond the bands themselves.

The band of suite 1 holds only 4.2 % of the hazard weight against 43.0 % for
suite 2, which quantifies the argument of the article in advance: suite 1 sits
in a thin tail of the deaggregation and should misrepresent the target the more
of the two. It does, its short-period spectral accelerations and :math:`ASI`
falling outside the bounds.


What is validated
-----------------

**The target, tightly.** Given the epsilon of the deaggregation, every
published median and dispersion is reproduced to a fraction of a percent, at
all fourteen periods and for :math:`SI`, :math:`ASI` and :math:`I_a`. This
validates the Boore and Atkinson predictions and the combination over ruptures
together.

**The target, as configured.** The medians reproduce the article to within
1.5 %, and the dispersion is bounded above by the published value.

**The duration difference, exactly.** The article takes
:math:`\rho(Ds_{595}, SA)` = 0, no correlation model having existed in 2010,
and its duration is the Abrahamson and Silva base model with the 5-95 %
dispersion and the duration ratio left at zero. djura applies both the
Bradley (2011) correlation and the ratio that converts the base 5-75 % model
into 5-95 %, so its median is about 2.2 times the published one. Setting the
two back to the choices of the article reproduces the published value exactly,
which pins the difference to those two choices and nothing else.

**The focal mechanism test.** Figure 4(b) is a post-hoc chi-squared comparison
of the mechanism distribution of suite 1 against the GCIM prediction, giving
:math:`p` = 0.2009. Using the published prediction and the mechanism of the
fifteen records of table II gives :math:`\chi^2` = 3.210 and
:math:`p` = 0.2009 on two degrees of freedom.


Running the case
----------------

.. code-block:: bash

   # assertions only
   pytest tests/rs/validations/test_case1_bradley2010.py -m validation

   # redraw figures 2 to 5 (requires matplotlib)
   pytest tests/rs/validations/test_case1_bradley2010.py -m validation --plot

   # tabulate the computed values against the published ones
   pytest tests/rs/validations/test_case1_bradley2010.py -m validation --report

Four figures are written: the conditional spectrum of figure 2, the
distributions of figure 3, and a suite selected in each causal band against the
target with its Kolmogorov-Smirnov bounds.


Limitations
-----------

* **The selected records cannot match the article's.** The algorithm draws
  random realizations from the conditional distribution, and the article
  selected from NGA-West1 where these runs use the bundled dataset.
* **The focal mechanism distribution is not reachable from the bundled
  database.** European shallow crustal records skew normal and reverse, so a
  New Zealand, strike-slip dominated mechanism mix cannot be assembled from
  them, and a suite selected here fails the chi-squared test. Mechanism is
  deliberately not screened during selection, the argument of the article being
  that GCIM needs no such screen.
* **The** :math:`ASI` **correlation is a later model** than the pre-publication
  form the article used, 0.587 against 0.61, which carries into the
  :math:`ASI` median.
* :math:`Sa(10\,\mathrm{s})` **is absent from the target.** The correlations of
  the period-independent measures with SA guard with
  ``0.01 <= period < 10``, excluding the endpoint.
* **The ground motion models are not validated here.** Every check takes their
  predictions as given; a systematic error would shift target and suite
  together and pass unnoticed.
