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

Input files
-----------

Each case ships one input file recording its complete configuration, so every
setting can be read in one place rather than assembled from the test module:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File in ``tests/rs/validations/assets``
     - Case
   * - ``bradley2010_input.json``
     - Case 1, conditioned on Sa(1.0 s) = 0.165 g, with the 1000 ruptures of
       the disaggregation inline
   * - ``bradley2012_input.json``
     - Case 2 at the 10 % in 50 year level with the weights of equation (16),
       with the 62 ruptures inline
   * - ``lin2013_input.json``
     - Case 3 conditioned on Sa(2.6 s) = 0.456 g, the mean deaggregation
       rupture

Each is the configuration of one sub-case; the others differ only in named
fields, the causal band and record count for case 1, the exceedance level and
weight vector for case 2, and the conditioning period for case 3.

One step is needed before such a file can be passed to
:class:`~djura.record_selection.gcim.GCIM`. ASI, SI and DSI are predicted by
*indirect* models, which derive their intensity measure from an underlying
spectral acceleration model and take that model as a constructed instance,
whose ``REQUIRES_*`` attributes they copy onto themselves. JSON can only carry
its name, so the name has to be resolved first:

.. code-block:: python

   import json

   from djura.record_selection.gcim import GCIM
   from djura.record_selection.gsim import models as gsim_models


   def resolve(name, **kwargs):
       """Instantiate a model by name, recursing into a nested 'gmpe'"""
       if isinstance(kwargs.get("gmpe"), str):
           kwargs = dict(kwargs, gmpe=resolve(kwargs["gmpe"]))

       return getattr(gsim_models, name)(**kwargs)


   data = json.loads(open("bradley2010_input.json").read())
   for entry in data["gmms"]:
       for spec in entry.values():
           for kwargs in spec.get("kwargs", []):
               if isinstance(kwargs.get("gmpe"), str):
                   kwargs["gmpe"] = resolve(kwargs["gmpe"])

   gcim = GCIM(data, conditional=True)
   gcim.create()
   gcim.select()

The test modules build the same dictionaries directly, which is why they do not
read these files.

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

Case 2 — Bradley (2012), Los Angeles
------------------------------------

   Bradley, B. A. (2012). A ground motion selection algorithm based on the
   generalized conditional intensity measure approach. *Soil Dynamics and
   Earthquake Engineering*, 40, 48-61. DOI:
   `10.1016/j.soildyn.2012.04.007 <https://doi.org/10.1016/j.soildyn.2012.04.007>`_

Site in Los Angeles, Vs30 = 760 m/s, conditioned on SA(3.0 s) at the 50, 10 and
1 per cent in 50 year exceedance levels. The intensity measure vector runs to
sixteen entries, spectral accelerations together with PGA, PGV, ASI, SI, DSI,
CAV and the two significant durations.

The result of the article is the effect of the weight vector. Weighting spectral
acceleration alone leaves the CAV and duration distributions of the selected
suite biased, and moving thirty per cent of the weight onto CAV and the two
durations removes the bias while matching the spectral ordinates just as well.
That contrast is what the module asserts, along with the target it is measured
against.

Why the target is not reproduced exactly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The article built its targets from the full disaggregated rupture set, and the
author confirms that a rupture-by-rupture listing was never archived, the
probabilities having been computed internally. The asset
``bradley2012_deagg_sa3pt0.json`` therefore stands in for it, recovered from the
bars of the published disaggregation: 62 weighted magnitude-distance cells per
level at bin centres of 10 km and 0.5 magnitude units.

That substitution has two bounded consequences, and neither is a defect in
djura.

The plotted disaggregation stops at 110 km, so the recovered bars carry 96.6,
99.2 and 99.8 per cent of the hazard at the three levels. The missing part is
the distant tail, and dropping it raises the medians and narrows the
distribution. The error tracks the truncation exactly, which is how it is
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
   * - Worst median, djura / article
     - 1.31
     - 1.10
     - 0.92

Collapsing each cell onto its centre removes the spread within it, which
accounts for a residual five to seven per cent dispersion deficit at the 1 per
cent level, where truncation is negligible. It touches only the
between-rupture term of the variance, so the measures most strongly correlated
with the conditioning one are untouched: DSI, SA(2.0 s) and SA(5.0 s) all
reproduce their dispersion to within half a per cent at every level.

Accordingly the medians are asserted, at fifteen per cent for the two rarer
levels and thirty-five per cent for the 50 per cent level, whose rupture set is
the most truncated; the dispersions are reported rather than asserted. Table 4
of the article, the correlations with SA(3.0 s), is reproduced to within 0.0005
and is reported too.

Component models and assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **SA(10.0 s)** is dropped from the article's vector of seventeen, so one
  published column is not checked. Every correlation of a period-independent
  measure with SA guards with ``0.01 <= period < 10``, excluding the endpoint.
  The weights are renormalised over the eight ordinates retained, which
  preserves the seventy-thirty split between the spectral and the other
  measures that the article studies.
* **CAV** is predicted by ``CampbellBozorgnia2008``, which carries the CB10 CAV
  equation the article cites.
* **Duration** runs six to seventeen per cent below the published median at
  every level. ``BommerEtAl2009RSD`` reads ztor, which the article does not
  state and which is assumed at 3 km.
* Reverse faulting and dip are assumed as well, neither being stated. dip is
  inert here: with the Joyner-Boore and rupture distances equal and ztor at
  least 1, the hanging-wall term of ``CampbellBozorgnia2008``, the only place
  dip enters, is zero.

Selection
~~~~~~~~~

No causal screening is applied, which is explicit in the article and central to
its argument, so the selection searches the whole database. The two suites of
thirty records are selected at the 10 per cent level, one per weight vector.

The contrast reproduces plainly. Against djura's own target median, the suite
selected on spectral acceleration alone overshoots CAV by 1.6, Ds575 by 2.6 and
Ds595 by 2.7, and the fuller vector brings all three back to within nine per
cent, while both suites match the spectral ordinates to within twelve per cent.
The article's second finding follows too: the fuller vector needs less
amplitude scaling, a median factor of 1.2 against 3.5, where the article reports
1.1 against 1.9.

The causal parameters of the suite are not comparable. The bundled database
holds few large-magnitude records where the article used NGA-West1, so the
algorithm reaches for smaller ones and scales harder, giving mean magnitudes
0.2 to 0.4 units below the published suites and wider distance dispersions. What
survives the change of database, and is asserted, is the contrast between the
two vectors.

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

What is asserted
~~~~~~~~~~~~~~~~

The conditional spectrum has properties that follow from its definition and
must hold whatever the database, and those are what the module checks: the
target is pinched at the conditioning period, its median equalling the
conditioning amplitude with no variability; the uniform hazard spectrum
envelopes it and touches it at that period; the target peaks there; and the
response spectra of the selected suite are pinched there too. The suite is also
compared against the conditional mean spectrum of figure 3, at the coarse
tolerance appropriate to values read off a logarithmic figure rather than
tabulated.

The one hazard value the article publishes, Sa at the first modal period at the
2 % in 50 year level, is asserted against the recovered deaggregation, which is
what validates the recovery procedure as a whole.

Databases
~~~~~~~~~

This case is parametrised over two prospective databases so that the influence
of the record set is isolated. One is bundled; the other must be supplied by
the reader and located with ``DJURA_VALIDATION_FLATFILE``, and that
parametrisation skips when the file is absent::

   export DJURA_VALIDATION_FLATFILE=/path/to/your/flatfile.pickle
   pytest tests/rs/validations/test_case3_lin2013.py -m validation

Deviations
~~~~~~~~~~

The article's lengthened period of 5.0 s is omitted throughout. The 2008 hazard
model provides no spectral acceleration above 3 s, so its hazard cannot be
deaggregated there and is not extrapolated.

The deaggregation was run at 360 m/s, the nearest site class the service
accepts, while the ground motion model is given the 400 m/s of the article. The
service rejects an arbitrary velocity, and 360 m/s is retained on the evidence
that it reproduces the one amplitude the article publishes to within 1.3 per
cent, where bracketing 400 m/s between two site classes is 6.6 per cent out.

With ``--plot``, two figures are written per database, the conditional spectra
of figure 2(b) and the selected suite of figure 3.

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
