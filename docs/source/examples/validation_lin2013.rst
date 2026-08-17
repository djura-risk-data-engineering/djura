Validation: conditional spectrum selection (Lin et al., 2013)
=============================================================

This page documents the replication of a published ground motion selection
case, and is intended both as evidence that the conditional spectrum machinery
behaves as its literature describes, and as a worked example of how to set up
a selection against an external seismic hazard analysis.

    Lin, T., Haselton, C. B., & Baker, J. W. (2013). Conditional spectrum-based
    ground motion selection. Part I: Hazard consistency for risk-based
    assessments. *Earthquake Engineering & Structural Dynamics*, 42(11),
    1847-1865. DOI: `10.1002/eqe.2301 <https://doi.org/10.1002/eqe.2301>`_

The conditional spectrum (CS) is the special case of the generalized
conditional intensity measure distribution in which the intensity measure
vector contains only spectral accelerations. The case is therefore replicated
by listing only ``SA`` entries in ``imi``, with uniform weights.

The case lives in
`tests/rs/validations/test_case3_lin2013.py <https://github.com/djura-risk-data-engineering/djura/blob/main/tests/rs/validations/test_case3_lin2013.py>`_.


The case
--------

A site in Palo Alto, California, with :math:`V_{s30}` = 400 m/s. The structure
is a twenty storey reinforced concrete special moment frame, the FEMA P695
Building 1020, with elastic modal periods of 2.6, 0.85 and 0.45 s and a
lengthened period of 5.0 s. Ground motions are selected at each of those
conditioning periods, forty per period, at the 2 % in 50 year exceedance
probability.

.. list-table::
   :header-rows: 1
   :widths: 12 14 20 54

   * - Sub-case
     - :math:`T^*`
     - Meaning
     - Status
   * - ``3A_T1``
     - 2.60 s
     - first modal period
     - replicated
   * - ``3B_T2``
     - 0.85 s
     - second modal period
     - replicated
   * - ``3C_T3``
     - 0.45 s
     - third modal period
     - replicated
   * - ``3D_2T1``
     - 5.00 s
     - lengthened period
     - not replicable, see `Limitations`_


Record databases
----------------

.. important::

   **The NGA-West2 database is not distributed with djura and is not
   downloaded by it.** It must be obtained separately by the user, converted
   to the metadata format djura expects, and supplied through the
   ``DJURA_METADATA_PATH`` environment variable.

   The dataset djura fetches on first use is the pan-European ESM flatfile
   (``flatfile_shallow.pickle``). Nothing in the package installs, downloads
   or redistributes NGA-West2, whose terms of use are set by PEER.

   See :doc:`../custom_metadata` for the expected schema.

The case is run twice, against one database at a time, so that the influence
of the prospective record set is isolated:

.. code-block:: bash

   # NGA-West2, supplied by the user
   export DJURA_METADATA_PATH=/path/to/NGA_W2_v2.pickle

   # ESM, downloaded by djura on first use
   export DJURA_METADATA_PATH=/path/to/flatfile_shallow.pickle

The article used NGA-West1, so neither run can reproduce its record list. The
NGA-West2 run is the nearer analogue; the ESM run is a deliberate sensitivity
test of a European database against Californian hazard.


Seismic hazard inputs
---------------------

The article reports only that mean deaggregation values of magnitude and
distance were taken from the USGS web tool, and tabulates no value other than
:math:`Sa(2.6\,\mathrm{s})` = 0.45 g. The scenarios were therefore recovered
by repeating that deaggregation against the USGS NSHMP service:

.. code-block:: text

   https://earthquake.usgs.gov/nshmp-haz-ws/deagg/
       {edition}/{region}/{longitude}/{latitude}/{imt}/{vs30}/{returnPeriod}

Requesting the endpoint without arguments returns a usage document listing the
permitted values. The arguments used were edition ``E2008``, the 2008
conterminous US model and the model era of the article; region ``WUS``;
longitude -122.1430 and latitude 37.4419; shear wave velocity 360 m/s; and
return period 2475 years. One request was issued per spectral period, for
example:

.. code-block:: bash

   curl "https://earthquake.usgs.gov/nshmp-haz-ws/deagg/E2008/WUS/-122.1430/37.4419/SA2P0/360/2475"

Reading the ``Deaggregation targets`` and ``Mean (over all sources)`` entries
of each response gives the uniform hazard spectrum and the mean causal
rupture:

.. list-table::
   :header-rows: 1
   :widths: 16 20 16 20 16

   * - :math:`T` [s]
     - :math:`Sa` [g]
     - :math:`\bar{M}`
     - :math:`\bar{R}` [km]
     - :math:`\bar{\varepsilon}`
   * - 0.30
     - 1.740
     - 7.35
     - 10.4
     - 1.62
   * - 0.50
     - 1.507
     - 7.48
     - 10.2
     - 1.57
   * - 0.75
     - 1.236
     - 7.60
     - 10.2
     - 1.50
   * - 1.00
     - 1.021
     - 7.66
     - 10.1
     - 1.45
   * - 2.00
     - 0.583
     - 7.79
     - 11.4
     - 1.33
   * - 3.00
     - 0.399
     - 7.84
     - 11.1
     - 1.24

None of the conditioning periods is a model period, so each quantity is
interpolated linearly in the logarithm of the period, which is the treatment
the article describes. This yields the scenarios actually used:

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 20 16

   * - :math:`T^*` [s]
     - :math:`Sa(T^*)` [g]
     - :math:`\bar{M}`
     - :math:`\bar{R}` [km]
     - :math:`\bar{\varepsilon}`
   * - 2.60
     - 0.456
     - 7.82
     - 11.2
     - 1.27
   * - 0.85
     - 1.138
     - 7.63
     - 10.2
     - 1.48
   * - 0.45
     - 1.552
     - 7.45
     - 10.3
     - 1.58

The interpolated :math:`Sa(2.6\,\mathrm{s})` of 0.456 g agrees with the
published 0.45 g to 1.3 %, which confirms the site, the model edition, the
return period and the remaining settings.

On the shear wave velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~

The service accepts only a discrete set of site classes and rejects 400 m/s
outright, so the deaggregation cannot be run at the velocity the article
states. Bracketing it, by deaggregating at 360 and 537 m/s and interpolating
in the logarithm of the velocity, gives :math:`Sa(2.6\,\mathrm{s})` = 0.420 g,
which is 6.6 % *below* the published value, whereas 360 m/s alone gives
0.456 g, 1.3 % above it.

The velocity of 360 m/s is therefore used for the deaggregation, on the
evidence that it reproduces the one value the article publishes; the article
most likely selected a site class in the hazard tool rather than an arbitrary
velocity. The ground motion model still receives the 400 m/s of the article.


Correlation model
-----------------

The article obtains the correlation between spectral accelerations from Baker
and Jayaram (2008). This has to be requested explicitly, because
``CORRELATION_MODELS`` lists another model first for the ``SA-SA`` pair and
the first entry is the default:

.. code-block:: json

   "correlation-models": {"SA-SA": "baker_jayaram"}

.. note::

   Any pair not named keeps its registered default. The models actually used
   are reported back in ``gcim.output_create["correlation_models"]``, so a
   result always states which equations produced it. Omitting this key changes
   the target materially: the conditional standard deviation at 10 s moves
   from 0.701 to 0.757.


Input file
----------

.. code-block:: json

   {
     "gmms": [
       {"SA": {"names": ["CampbellBozorgnia2008"], "weights": [1]}}
     ],
     "correlation-models": {"SA-SA": "baker_jayaram"},
     "site-parameters": {
       "vs30": 400,
       "z2pt5": 2.0,
       "mechanism": "strike-slip fault"
     },
     "ruptures": [
       {"mag": 7.82, "rjb": 11.2, "rrup": 11.2, "ztor": 0.0,
        "d_hyp": 4.7, "weight": 1.0}
     ],
     "imi": [
       "SA(0.1s)", "SA(0.2s)", "SA(0.3s)", "SA(0.45s)", "SA(0.5s)",
       "SA(0.75s)", "SA(0.85s)", "SA(1.0s)", "SA(1.5s)", "SA(2.0s)",
       "SA(2.6s)", "SA(3.0s)", "SA(4.0s)", "SA(5.0s)", "SA(7.5s)",
       "SA(10.0s)"
     ],
     "im-star": {
       "type": "SA(2.6)",
       "value": 0.456,
       "gmms": {"names": ["CampbellBozorgnia2008"], "weights": [1]}
     },
     "num-components": 1,
     "component-definition": "geomean",
     "num_records": 40,
     "nreplicate": 1,
     "seed": 1,
     "ks_alpha": 0.1,
     "max_scaling_factor": 10,
     "context_limits": {},
     "im_weights": []
   }

All four conditioning periods are kept in ``imi`` whichever one is conditioned
upon, so that the spectra of every suite can be compared at each of them. No
causal parameter screening is applied, matching the article.


What is validated
-----------------

The selected records cannot match the article's, since the algorithm draws
random realizations from the conditional distribution and the databases
differ. What is validated is the target distribution, the properties the
article states, and the statistics of the suite.

**Against the conditional spectrum equations.** Equations (1) to (3) of the
article are evaluated independently from the ground motion model predictions
held in the creation output, and compared with the target:

.. math::

   \varepsilon(T^*) &= \frac{\ln Sa(T^*) - \mu(T^*)}{\sigma(T^*)} \\
   \mu(T_i \mid T^*) &= \mu(T_i)
       + \rho(T_i, T^*)\,\varepsilon(T^*)\,\sigma(T_i) \\
   \sigma(T_i \mid T^*) &= \sigma(T_i)\sqrt{1 - \rho^2(T_i, T^*)}

Agreement is to :math:`10^{-6}` at every period. This validates the assembly
of the conditional distribution, taking the ground motion model and the
correlation equations as given.

**Against the published hazard.** The recovered
:math:`Sa(2.6\,\mathrm{s})` matches the published 0.45 g to 1.3 %.

**Against figure 2(b).** The article states that the uniform hazard spectrum
envelopes the conditional mean spectra, and that each equals it at its own
conditioning period. Both hold exactly: the ratio is 1.000 at :math:`T^*` and
falls monotonically with period separation, to 0.43 at the far ends. Each
spectrum also peaks, relative to the uniform hazard spectrum, at its own
conditioning period.

**Against figure 3.** The suite median is compared with the conditional mean
spectrum read from that figure. Because the published curve lies within a band
of forty record spectra in a raster figure, those readings carry an
uncertainty of roughly a quarter, and the comparison establishes the level of
the suite rather than its detail.

.. list-table:: Suite against target and figure 3, NGA-West2, :math:`T^*` = 2.6 s
   :header-rows: 1
   :widths: 12 18 16 16 19 19

   * - :math:`T` [s]
     - Figure 3 [g]
     - Target [g]
     - Suite [g]
     - Target :math:`\sigma`
     - Suite :math:`\sigma`
   * - 0.10
     - 0.500
     - 0.535
     - 0.531
     - 0.541
     - 0.556
   * - 0.50
     - 0.750
     - 0.782
     - 0.785
     - 0.524
     - 0.526
   * - 1.00
     - 0.650
     - 0.708
     - 0.722
     - 0.469
     - 0.471
   * - 2.60
     - 0.450
     - 0.456
     - 0.456
     - 0.000
     - 0.000
   * - 5.00
     - 0.200
     - 0.213
     - 0.214
     - 0.453
     - 0.456
   * - 10.00
     - 0.100
     - 0.096
     - 0.096
     - 0.701
     - 0.699

The suite tracks the target to within about 2 % in median and 3 % in
dispersion at all sixteen periods, and both lie within about 10 % of the
figure readings. The dispersion column is the more searching comparison: it
rises from zero at the conditioning period to 0.70 at 10 s, and the suite
reproduces that, not merely the median.


Running the case
----------------

.. code-block:: bash

   # assertions only
   pytest tests/rs/validations

   # redraw figures 2(b) and 3 from the results (requires matplotlib)
   pytest tests/rs/validations --plot

   # tabulate the computed values against the published ones
   pytest tests/rs/validations --report

   # both, written elsewhere
   pytest tests/rs/validations --plot --report --plot-dir /path/to/output

Figures and tables are written to ``tests/rs/validations/_plots`` by default,
one of each per database. Neither option is enabled by default, and matplotlib
is not a dependency of the package.


Limitations
-----------

* **The 5.0 s conditioning period cannot be replicated.** The 2008 model
  provides no spectral acceleration above 3 s, the service rejecting ``SA4P0``
  and ``SA5P0`` for the western US, so its hazard cannot be deaggregated. It
  is not extrapolated.
* **The rupture scenarios are reconstructed, not published.** They come from
  the deaggregation above, not from the article, which tabulates none.
* **The ground motion model is not validated here.** Every check takes the
  Campbell and Bozorgnia (2008) predictions as given; a systematic error in
  them would shift target and suite together and pass unnoticed.
* **The hazard consistency of figures 4(c) to 4(e) is not tested.** That is
  the central result of the article, and it requires the ten intensity levels
  and the site hazard curves.
* **An approximate conditional spectrum is used**, built from a single mean
  magnitude and distance and a single ground motion model, as in the article.
  Section 3.2 of the article notes that this underestimates the conditional
  standard deviation at periods far from :math:`T^*`.
