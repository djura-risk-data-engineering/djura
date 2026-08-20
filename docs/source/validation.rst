Validation against published case studies
=========================================

djura's record-selection routines are checked against the published results of
the articles that introduced the methods they implement. Each case is a pytest
module under ``tests/rs/validations``, and each one reproduces the numbers a
reader can look up in the article, not merely a value recorded from an earlier
djura run. The pages below describe what is compared, what agreement is
expected, and how to run the cases locally.

The cases are excluded from continuous integration: they build targets over
hundreds of rupture scenarios and search the full record database, so they take
minutes rather than seconds. They are run on request.

Running the cases
-----------------

The validation cases live in the repository, not in the installed wheel, so
start from a source checkout::

   git clone https://github.com/<your-fork>/djura-pypi
   cd djura-pypi

Install the record-selection application, and the plotting extra if the figures
are wanted::

   pip install -e ".[record_selection,plot]"

Every test carries the ``validation`` marker, so the whole suite is::

   pytest -m validation

and one case at a time::

   pytest tests/rs/validations/test_case1_bradley2010.py -m validation

The marker composes with the ``slow`` marker that these tests also carry, which
is what keeps them out of CI:

.. code-block:: sh

   pytest -m "not slow and not validation"   # what CI runs
   pytest -m validation                      # the validation cases only

Options
-------

Two options are registered by ``tests/rs/validations/conftest.py``.

``--report``
   Write a Markdown table of every computed quantity against its published
   counterpart. Use this to inspect agreement quantity by quantity rather than
   through pass or fail.

``--plot``
   Redraw the figures of the articles, djura's result over the published one.
   Requires the ``plot`` extra; the tests skip with a message if matplotlib is
   absent.

``--plot-dir``
   Where to write both, by default ``tests/rs/validations/_plots``.

For example::

   pytest -m validation --report --plot --plot-dir /tmp/djura-validation

Data the cases need
-------------------

Two things are required, and both are already in place for the case below.

The **record database** is the bundled dataset,
``flatfile_shallow_v1.pickle``, which is downloaded automatically on first use
(see :doc:`dataset`). A case skips with a message if the file it needs is
absent. Each case sets ``DJURA_METADATA_PATH`` itself for the duration of the
run and restores whatever was there before, so no environment setup is needed
and an existing value is not disturbed.

The **hazard input** of a case is either tabulated in the module or, where it is
too large for that, held as an asset next to it under
``tests/rs/validations/assets``. Where a case can also be run against a database
that is not distributed with djura, it reads the path from an environment
variable and skips that parametrisation when it is unset. Case 3 does this with
``DJURA_VALIDATION_FLATFILE``.

Case 1 — Bradley (2010), Christchurch
-------------------------------------

   Bradley, B. A. (2010). A generalized conditional intensity measure approach
   and holistic ground-motion selection. *Earthquake Engineering & Structural
   Dynamics*, 39(12), 1321-1342. DOI:
   `10.1002/eqe.995 <https://doi.org/10.1002/eqe.995>`_

Rock site in Christchurch, New Zealand, Vs30 = 760 m/s, conditioned on
Sa(1.0 s) = 0.165 g, the 2 % in 50 year value. The intensity measure vector is
``{SA, SI, ASI, Ia, Ds595}``, which is what makes the case worth replicating:
it exercises the period-independent measures and their correlations, not only
spectral accelerations.

The published target of figures 2 and 3 is tabulated in the module as
``PUBLISHED_SA`` and ``PUBLISHED_IMS``, and the disaggregation behind figure 1
is the asset ``bradley2010_deagg_sa1p0_2pct50.csv``: 1000 rupture scenarios
with magnitude, source-to-site distance, epsilon, rake and percent
contribution. Table II of the article, both suites with their causal
parameters, and the focal-mechanism target of figure 4(b) are in
``bradley2010_suites.json``.

What is asserted
~~~~~~~~~~~~~~~~

Two comparisons, for a reason that is worth understanding before reading the
results.

The article takes epsilon per rupture from the disaggregation itself, which the
NZ national seismic hazard model produced with its own ground motion model,
while predicting the intensity measures with Boore and Atkinson (2008). djura
derives epsilon from the model it is given for the conditioning intensity
measure, since it carries neither the NZ model nor an input for supplying
epsilon directly. Either is a reasonable choice; they simply differ, and the
difference is confined to the dispersion at periods above the conditioning
period.

``test_target_is_exact_given_the_published_epsilon``
   Substitutes the epsilon of the disaggregation into djura's own per-rupture
   moments, leaving only its ground motion model predictions and its
   combination over the 1000 ruptures. Every published median and dispersion is
   reproduced to a fraction of a percent.

``test_target_matches_the_published_target``
   djura as configured. The medians must reproduce the article; the dispersion
   is required only not to exceed the published value, self-consistent epsilon
   being able to remove between-rupture variance but not add it.

Expected agreement
~~~~~~~~~~~~~~~~~~

With the published epsilon substituted, on the ESM database:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Quantity
     - Median, djura / article
     - Dispersion, djura / article
   * - SA, all 14 periods
     - 0.994 to 1.001
     - 1.000
   * - SI
     - 1.004
     - 1.007
   * - ASI
     - 0.984
     - 1.010
   * - Ia
     - 1.012
     - 1.004
   * - Ds595, article's own choices restored
     - 1.0000
     - 1.0000

The correlations of table I are reported rather than asserted, being published
to two decimals: djura gives 0.4157, 0.7490, 0.9160, 0.5866 and 0.6970 against
the published 0.42, 0.75, 0.92, 0.61 and 0.7.

Component models of the period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The case is from 2010 and several component models have since been superseded,
so exact reproduction needs the models of the time rather than their
replacements. Three differences remain, all understood and all documented in
the module:

* **ASI.** The article used a pre-publication ASI-SA correlation, later
  published as Bradley (2011), which is what djura implements: 0.587 against
  the 0.61 of table I. This carries into the ASI median, which is why that one
  panel is held to a looser tolerance.
* **Ds595.** The article takes rho(Ds595, SA) = 0, no correlation model having
  existed in 2010, and its duration is the Abrahamson and Silva base model with
  the 5-95 % dispersion and the duration ratio left at zero. djura applies both
  the Bradley (2011) correlation and the ratio that converts the base 5-75 %
  model into 5-95 %, so its median is about 2.2 times the published one.
  Setting the two back to the choices of the article reproduces the published
  value exactly, which is what the test asserts.
* **SA(10 s)** is absent from the target. djura's SI-SA and related correlation
  functions guard with ``0.01 <= period < 10``, excluding the endpoint.

Several of the correlation pairs needed here have more than one model
registered in djura, so the ones the article used are named explicitly through
``correlation-models``. With those supplied, table I is reproduced.

Selection
~~~~~~~~~

The article's two suites were screened on causal parameters, magnitude below 6
and distance below 20 km for suite 1, and above 7 and 50 km for suite 2. The
figures redraw a suite selected within each band and compare its distribution
against the target with djura's own Kolmogorov-Smirnov bounds at the 10 %
significance level.

The selected records are not expected to match the article's. The algorithm
draws random realizations from the conditional distribution, so the suite
depends on the seed and on the realization draw, and the prospective database
is ESM where the article used NGA-West1. What is comparable is the suite
statistics: on ESM the mean causal magnitude and distance fall within 0.26
magnitude units and 19 km of the published suites, with no screening beyond the
causal bands themselves.

One consequence of the database is worth recording. Figure 4(b) of the article
is a post-hoc chi-squared test of the focal mechanism distribution of suite 1
against the GCIM prediction, giving p = 0.2009; the published values in
``bradley2010_suites.json`` reproduce that exactly. A suite selected from ESM
does not: European shallow crustal records skew normal and reverse, so a New
Zealand, strike-slip dominated mechanism mix is not reachable from that
database. This is a limit of the prospective database, not of the method, and
mechanism is deliberately not screened during selection — the argument of the
article is that GCIM needs no such screen.

Figures
~~~~~~~

With ``--plot``, four figures are written to the plot directory.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Content
   * - ``case1_figure2.png``
     - Conditional distribution of Sa, median and 16th and 84th percentiles,
       against figure 2
   * - ``case1_figure3.png``
     - Conditional distribution of each intensity measure, against figure 3
   * - ``case1_figure4.png``
     - A suite selected in the causal band of suite 1, against the target,
       with Kolmogorov-Smirnov bounds
   * - ``case1_figure5.png``
     - The same for the band of suite 2

djura is drawn solid and the published values dashed throughout. The ASI and
Ds595 panels carry a caption naming the component model that differs, so that
the offset reads as documented rather than as a fault.

Case 3 — Lin, Haselton and Baker (2013), Palo Alto
--------------------------------------------------

   Lin, T., Haselton, C. B., & Baker, J. W. (2013). Conditional spectrum-based
   ground motion selection. Part I: Hazard consistency for risk-based
   assessments. *Earthquake Engineering & Structural Dynamics*, 42(11),
   1847-1865. DOI:
   `10.1002/eqe.2301 <https://doi.org/10.1002/eqe.2301>`_

The conditional spectrum is the special case of the GCIM distribution in which
the intensity measure vector contains only spectral accelerations, so the case
is replicated by listing only SA entries with uniform weights.

The article tabulates almost none of its hazard input, so the module documents
how the deaggregation was recovered from the USGS service, argument by
argument, and records the check that validates the recovery: the interpolated
Sa(2.6 s) of 0.456 g against the 0.45 g the article publishes. The procedure
can be repeated as given.

This case is parametrised over two prospective databases so that the influence
of the record set is isolated. One is bundled; the other must be supplied by
the reader and located with ``DJURA_VALIDATION_FLATFILE``, and that
parametrisation skips when the file is absent::

   export DJURA_VALIDATION_FLATFILE=/path/to/your/flatfile.pickle
   pytest tests/rs/validations/test_case3_lin2013.py -m validation

Adding a case
-------------

A new case is a module under ``tests/rs/validations`` following the same
pattern. In order of importance:

#. **Compare against the article, not against djura.** Published values belong
   in the module as named constants, or as an asset when there are too many for
   that, with the table or figure they come from named in a comment.
#. **Say what cannot match, and why.** Superseded component models, a different
   prospective database, and the randomness of the selection all produce real
   differences. Record each one in the module docstring, and hold the affected
   quantity to a tolerance that reflects the size of the known difference rather
   than dropping the assertion.
#. **Assert the output, report the rest.** Assert the target and the suite
   statistics. Quantities that are informative but not conclusions of the
   article — intermediate correlations, epsilon populations — belong in the
   ``--report`` table.
#. **Skip, do not fail, on missing data.** A case whose database or hazard
   input is unavailable must skip with a message naming what to provide.
#. **Carry both markers**, ``validation`` at module level through
   ``pytestmark`` and ``slow`` on the test classes, so the case stays out of CI.
