Verification against OpenQuake
==============================

Most ground motion models (GMMs) in ``djura.record_selection.gsim.models`` are
adapted from the `OpenQuake Engine <https://github.com/gem/oq-engine>`_
version 3.23.0. OpenQuake verifies each of its GMMs against tables of expected
medians and standard deviations. djura can be checked against the same tables,
which shows that the port reproduces OpenQuake for the models you rely on.

The tables are not shipped with djura. They are downloaded from OpenQuake's
``v3.23.0`` tag the first time they are needed and cached under
``~/.cache/djura/oq-gsim-tables/``. OpenQuake itself is not required.

Running the verification
------------------------

The verification is a test module in the source repository, so start from a
source checkout with the record-selection extra installed::

   git clone https://github.com/djura-risk-data-engineering/djura
   cd djura
   pip install -e ".[record_selection]" pytest

Name the GMMs to verify with ``--gsim``, as a comma-separated list of djura
class names::

   pytest tests/rs/gsim_verification --gsim BooreEtAl2014,ChiouYoungs2014

Only the tables of those GMMs are downloaded and checked. Without ``--gsim``
nothing is run, and an unknown name stops the run with an error. Each
OpenQuake check is reported as its own test case, for example
``BooreEtAl2014{'region': 'CAL', 'sof': False}-BSSA_2014_CALIFORNIA_NOSOF_MEAN``.
The GMMs that can be named are the ``gsim`` entries of
``tests/rs/gsim_verification/oq_gsim_tables.json``.

The verification is not part of continuous integration.

What is compared
----------------

Each entry of ``oq_gsim_tables.json`` is one check from OpenQuake's own test
suite: the GMM, the arguments it is built with, its tables and OpenQuake's
tolerances for that check. The comparison follows OpenQuake's:

* The median is compared as ``exp(mean)``, and each standard deviation (total,
  inter-event, intra-event) as it is. A standard deviation is only compared
  when the GMM declares it.
* The discrepancy is ``|djura / OpenQuake - 1|`` in per cent, and every value
  of every table must be within the tolerance OpenQuake uses for that check.
* Rows sharing a rupture are evaluated together. A missing rake is taken as
  zero, and ``rrup`` falls back on the distance the table provides, as in
  OpenQuake.

A GMM written for one scenario at a time, which is how djura calls GMMs during
selection, is evaluated row by row.

Coverage
--------

With OpenQuake 3.23.0, 195 checks over 87 djura GMMs are in the manifest, and
all of them pass.

Three adjustments are made where djura differs from OpenQuake by design:

* **CampbellBozorgnia2019, CAV.** djura reports CAV in g·s, like the other CAV
  models of the package, where OpenQuake reports m/s. The OpenQuake median is
  divided by the standard gravity before comparing.
* **ChiouYoungs2014 regions.** OpenQuake selects the region with an argument;
  djura with the ``ChiouYoungs2014Japan``, ``ChiouYoungs2014Italy`` and
  ``ChiouYoungs2014Wenchuan`` classes, which those checks use.
* **CampbellBozorgnia2014 and 2019.** OpenQuake passes its high-Q and low-Q
  coefficient tables and the Japan flag as arguments; djura has the ``HighQ``,
  ``LowQ`` and ``JapanSite`` classes, which those checks use.

The following GMMs have no check of their own:

* Classes that only fix an argument of a verified class: the regional
  ``AbrahamsonEtAl2014Reg*`` and ``AbrahamsonGulerce2020SInter*`` and
  ``SSlab*`` classes, and the basin and no-style-of-faulting ``BooreEtAl2014*``
  classes. The verified class is checked with the same arguments.
* GMMs with no OpenQuake table: ``AbrahamsonSilva1996``,
  ``AtkinsonBoore2006SGS``, the ``AtkinsonBoore2006*NSHMP2008`` classes,
  ``DavalosEtAl2020``, and the indirect models ``BradleyEtAl2009SI``,
  ``Bradley2010ASI``, ``Bradley2011DSI`` and ``GmpeIndirectAvgSA``.
* Two ``KothaEtAl2020ESHM20`` checks, with the fast-attenuation ``c3`` and the
  high ``dl2l`` tables, whose inputs OpenQuake builds at run time. The other
  ``KothaEtAl2020ESHM20`` checks are included.

Updating the manifest
---------------------

The manifest is generated from OpenQuake's test modules, which are parsed and
never executed::

   python tests/rs/gsim_verification/generate_manifest.py --tag v3.23.0

The script prints the checks it could not follow, so they can be reviewed
when moving to another OpenQuake version.
