Validation against published case studies
=========================================

djura's record-selection routines are checked against the published results of
the articles that introduced the methods they implement. Each case is a pytest
module under ``tests/rs/validations`` and reproduces numbers a reader can look
up in the article, not values recorded from an earlier djura run. The pages
below give, for each case, the inputs and where they came from, what is
asserted, the agreement to expect, and what cannot be reproduced and why.

.. toctree::
   :maxdepth: 1

   bradley2010
   bradley2012
   lin2013

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Case
     - Article
     - What it exercises
   * - :doc:`bradley2010`
     - Bradley (2010), Christchurch
     - The full GCIM vector: spectral accelerations together with SI, ASI,
       Arias intensity and significant duration, and the correlations between
       them
   * - :doc:`bradley2012`
     - Bradley (2012), Los Angeles
     - The selection algorithm and the effect of the weight vector, over three
       exceedance levels and sixteen intensity measures
   * - :doc:`lin2013`
     - Lin, Haselton and Baker (2013), Palo Alto
     - The conditional spectrum, being the special case in which the vector
       holds only spectral accelerations

Running the cases
-----------------

The cases live in the repository, not in the installed wheel, so start from a
source checkout::

   git clone https://github.com/djura-risk-data-engineering/djura
   cd djura

Install the record-selection application, and the plotting extra if the figures
are wanted::

   pip install -e ".[record_selection,plot]"

Every test carries the ``validation`` marker, so the whole suite is::

   pytest -m validation

and one case at a time::

   pytest tests/rs/validations/test_case1_bradley2010.py -m validation

The cases are excluded from continuous integration: they build targets over
hundreds of rupture scenarios and search the full record database, so they take
minutes rather than seconds. The marker composes with the ``slow`` marker that
these tests also carry:

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

The **record database** is the bundled dataset, which is downloaded
automatically on first use; see :doc:`../../dataset`. A case skips with a
message if the database it needs is absent. Each case names it for the duration
of the run and restores whatever ``DJURA_METADATA_PATH`` held before, so no
environment setup is needed and an existing value is not disturbed. Where a
case can also be run against a database that is not distributed with djura, it
reads the path from an environment variable and skips that parametrisation when
it is unset.

The **hazard input** of a case is either tabulated in the module or, where it is
too large for that, held as an asset under ``tests/rs/validations/assets``.

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
