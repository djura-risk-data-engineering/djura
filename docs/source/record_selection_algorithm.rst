Record selection algorithm
==========================

This page describes how :class:`~djura.record_selection.gcim.GCIM` builds the
target distribution and selects records from it. The inputs are the rupture
scenarios with their probabilities, the ground motion models (GMMs) with their
logic-tree weights, the intensity measures (IMs) of interest, the conditioning
IM and the correlation models.

Selection procedure
-------------------

Given those inputs, the software first computes the mean and variance of each
IM for every pair of rupture scenario and GMM. Over multiple rupture scenarios
and GMMs, the target distribution of each IM is the mixture of these lognormal
distributions, each weighted by the product of the rupture probability and the
GMM weight. Its mean and variance are computed exactly by Equations
:eq:`cmf` and :eq:`cvf`, and the covariance matrix is built from them and the
conditional correlations, as described in `Simulating the target`_. Following
Jayaram et al. (2011), the resulting mean vector and covariance matrix are used
to simulate realisations of the target from a multivariate lognormal
distribution. Several sets of realisations may be generated, in which case the
set whose mean, standard deviation and skewness best match the target is
retained before any record is selected, as in Baker and Lee (2017). Each
realisation of this set is then assigned the record, optionally
amplitude-scaled, that satisfies all specified conditions and has the smallest
weighted sum of squared residuals in log space. This initial suite is improved
by the greedy optimisation of Jayaram et al. (2011), which replaces records one
at a time to improve the match between the sample mean and standard deviation
of the suite and those of the target, and which can be repeated until these
errors are within a tolerance, as in Baker and Lee (2017).

This procedure differs from the selection algorithm of Bradley (2012) in two
main respects. There, each realisation is drawn for a rupture scenario sampled
from the disaggregation, i.e., from the full generalised conditional intensity
measure (GCIM) mixture rather than from a single lognormal distribution, and
the choice among replicates is made on complete record suites, using the
Kolmogorov-Smirnov statistic against that mixture, rather than on the
realisations before any record is matched.

The algorithm is implemented using Numba (Lam et al., 2015) in Python to ensure
accelerated and efficient ground motion selection. Each selected record is
scaled, and the suite is returned with its scaling factors and goodness-of-fit
statistics, available on the returned object.

Unconditional target
--------------------

For an approximate case with a single rupture scenario and GMM, the
unconditional mean and variance are given by Equations :eq:`um1` and
:eq:`uv1`.

.. math::
   :label: um1

   \mu_{\ln IM_i} = \mu_{\ln IM_i^{g}\mid rup_r}

.. math::
   :label: uv1

   \sigma^2_{\ln IM_i} = \sigma^2_{\ln IM_i^{g}\mid rup_r}

where :math:`\mu_{\ln IM_i^{g}\mid rup_r}` and
:math:`\sigma^2_{\ln IM_i^{g}\mid rup_r}` are the mean and variance of
:math:`\ln IM_i` for a single rupture :math:`rup_r` characterised using GMM
:math:`g`. Generalising to multiple rupture scenarios and GMMs gives the
unconditional mean and variance in Equations :eq:`um2` and :eq:`uv2`.

.. math::
   :label: um2

   \mu_{\ln IM_i} = \sum_{r=1}^{n_{rup}} \sum_{g=1}^{n_{gmm,r}} p_r\, w_{r,g}\,
       \mu_{\ln IM_i^{g}\mid rup_r}

.. math::
   :label: uv2

   \sigma^2_{\ln IM_i} = \sum_{r=1}^{n_{rup}} \sum_{g=1}^{n_{gmm,r}} p_r\, w_{r,g}
       \left[ \sigma^2_{\ln IM_i^{g}\mid rup_r}
       + \left( \mu_{\ln IM_i} - \mu_{\ln IM_i^{g}\mid rup_r} \right)^2 \right]

where :math:`n_{rup}` is the number of rupture scenarios and
:math:`n_{gmm,r}` is the number of GMMs for rupture :math:`rup_r`;
:math:`p_r` is the probability of rupture :math:`rup_r`, with
:math:`\sum_{r=1}^{n_{rup}} p_r = 1`; and :math:`w_{r,g}` is the weight of GMM
:math:`gmm_g` for that rupture, with
:math:`\sum_{g=1}^{n_{gmm,r}} w_{r,g} = 1`. The squared term
:math:`\left(\mu_{\ln IM_i} - \mu_{\ln IM_i^{g}\mid rup_r}\right)^2` in
Equation :eq:`uv2` accounts for the between-branch dispersion, that is, the
spread of the individual branch means about the combined mean
:math:`\mu_{\ln IM_i}`. The logic-tree weights act as prior weights, whereas
the disaggregation weights act as posterior weights and are usually expressed
as a single product.

Conditional target
------------------

When conditioned on :math:`IM^*`, the moments become
:math:`\mu_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r}` and
:math:`\sigma^2_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r}`, where
:math:`\ln IM^*_h` is obtained from another GMM, denoted :math:`h`. For a given
rupture scenario :math:`rup_r` and GMM pair, the conditional mean and variance
are given by Equations :eq:`cm1` to :eq:`ceps`.

.. math::
   :label: cm1

   \mu_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r} = \mu_{\ln IM_i^{g}\mid rup_r}
       + \rho_{\ln IM_i^{g},\, \ln IM^*_h \mid rup_r}\,
       \sigma_{\ln IM_i^{g}\mid rup_r}\, \epsilon_{\ln IM^*_h \mid rup_r}

.. math::
   :label: cv1

   \sigma^2_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r} =
       \sigma^2_{\ln IM_i^{g}\mid rup_r}
       \left(1 - \rho^2_{\ln IM_i^{g},\, \ln IM^*_h \mid rup_r}\right)

.. math::
   :label: ceps

   \epsilon_{\ln IM^*_h \mid rup_r} =
       \frac{\ln IM^* - \mu_{\ln IM^*_h \mid rup_r}}{\sigma_{\ln IM^*_h \mid rup_r}}

Here, :math:`\rho` is the correlation between the residuals :math:`\epsilon`
of :math:`\ln IM_i^{g}` and :math:`\ln IM^*_h` (Baker, 2011), generalised to
two GMMs, :math:`g` and :math:`h`, for a rupture :math:`rup_r`. Baker and
Bradley (2017) investigated the dependence on rupture parameters and GMMs of
these correlation models, concluding that they could be treated as
independent. More recent work by Aristeidou et al. (2025) did notice a
non-negligible impact of these parameters, but for practical implementation
concluded that it could be omitted without adverse impact, allowing Equations
:eq:`cm1` and :eq:`cv1` to be written as Equations :eq:`cm2` and :eq:`cv2`.

.. math::
   :label: cm2

   \mu_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r} = \mu_{\ln IM_i^{g}\mid rup_r}
       + \rho_{\ln IM_i,\, \ln IM^*}\, \sigma_{\ln IM_i^{g}\mid rup_r}\,
       \epsilon_{\ln IM^*_h \mid rup_r}

.. math::
   :label: cv2

   \sigma^2_{\ln IM_i^{g}\mid \ln IM^*, rup_r} =
       \sigma^2_{\ln IM_i^{g}\mid rup_r}\left(1 - \rho^2_{\ln IM_i,\, \ln IM^*}\right)

The combined conditional mean is the weighted sum over the GMMs, given in
Equation :eq:`ccomb`, with the weights :math:`w_h` taken from the logic-tree
branch for :math:`rup_r` or derived from the disaggregation.

.. math::
   :label: ccomb

   \mu_{\ln IM_i^{g}\mid \ln IM^*, rup_r} =
       \sum_{h=1}^{n_{gmm,r}} w_h\, \mu_{\ln IM_i^{g}\mid \ln IM^*_h, rup_r}

For a single rupture scenario, the conditional mean and variance follow from
Equations :eq:`csm` and :eq:`csv`.

.. math::
   :label: csm

   \mu_{\ln IM_i \mid \ln IM^*} = \mu_{\ln IM_i^{g}\mid \ln IM^*, rup_r}

.. math::
   :label: csv

   \sigma^2_{\ln IM_i \mid \ln IM^*} = \sigma^2_{\ln IM_i^{g}\mid \ln IM^*, rup_r}

Extending to multiple rupture scenarios and GMMs gives the conditional mean and
variance in Equations :eq:`cmf` and :eq:`cvf`. This is the full expansion of
the GCIM approach (Bradley, 2010), with the addition here of multiple rupture
scenarios relative to the single scenario in the original formulation.

.. math::
   :label: cmf

   \mu_{\ln IM_i \mid \ln IM^*} = \sum_{r=1}^{n_{rup}} \sum_{g=1}^{n_{gmm,r}}
       p_r\, w_{r,g}\, \mu_{\ln IM_i^{g}\mid \ln IM^*, rup_r}

.. math::
   :label: cvf

   \sigma^2_{\ln IM_i \mid \ln IM^*} = \sum_{r=1}^{n_{rup}} \sum_{g=1}^{n_{gmm,r}}
       p_r\, w_{r,g} \left[ \sigma^2_{\ln IM_i^{g}\mid \ln IM^*, rup_r}
       + \left(\mu_{\ln IM_i \mid \ln IM^*}
       - \mu_{\ln IM_i^{g}\mid \ln IM^*, rup_r}\right)^2 \right]

Simulating the target
---------------------

The conditional distribution of the intensity measure vector
:math:`\ln \mathbf{IM}`, given the conditioning intensity measure
:math:`\ln IM^*`, is modelled as a multivariate normal distribution:

.. math::

   \ln \mathbf{IM} \mid \ln IM^* \sim \mathcal{N}\!\left(
       \boldsymbol{\mu}_{\ln \mathbf{IM} \mid \ln IM^*},\;
       \boldsymbol{\Sigma}_{\ln \mathbf{IM} \mid \ln IM^*}\right)

which is fully characterised by its mean vector :math:`\boldsymbol{\mu}` and
covariance matrix :math:`\boldsymbol{\Sigma}`, each assembled from the
conditional statistics of the individual :math:`\ln IM_i` terms. The
conditional mean and standard deviation of each :math:`\ln IM_i` enter the
distribution through the vectors described by Equation :eq:`vecs`:

.. math::
   :label: vecs

   \boldsymbol{\mu}_{\ln \mathbf{IM} \mid \ln IM^*} =
       \begin{bmatrix} \vdots \\ \mu_{\ln IM_i \mid \ln IM^*} \\ \vdots \end{bmatrix},
   \qquad
   \boldsymbol{\sigma}_{\ln \mathbf{IM} \mid \ln IM^*} =
       \begin{bmatrix} \vdots \\ \sigma_{\ln IM_i \mid \ln IM^*} \\ \vdots \end{bmatrix}

The covariance matrix :math:`\boldsymbol{\Sigma}_{\ln \mathbf{IM} \mid \ln IM^*}`
is assembled from the conditional correlations between each pair
:math:`(\ln IM_i,\, \ln IM_j)`, obtained by Equation :eq:`ccorr`.

.. math::
   :label: ccorr

   \rho_{\ln IM_i, \ln IM_j \mid \ln IM^*} =
       \frac{\rho_{\ln IM_i, \ln IM_j}
             - \rho_{\ln IM_i, \ln IM^*}\, \rho_{\ln IM_j, \ln IM^*}}
            {\sqrt{1 - \rho^2_{\ln IM_i, \ln IM^*}}\,
             \sqrt{1 - \rho^2_{\ln IM_j, \ln IM^*}}}

Each entry of :math:`\boldsymbol{\Sigma}` is then given by Equation
:eq:`ccov`.

.. math::
   :label: ccov

   \Sigma_{\ln IM_i, \ln IM_j \mid \ln IM^*} =
       \rho_{\ln IM_i, \ln IM_j \mid \ln IM^*}\,
       \sigma_{\ln IM_i \mid \ln IM^*}\, \sigma_{\ln IM_j \mid \ln IM^*}

yielding the full covariance matrix:

.. math::
   :label: cmat

   \boldsymbol{\Sigma}_{\ln \mathbf{IM} \mid \ln IM^*} =
   \begin{bmatrix}
   \sigma_{1|*}^2 & \rho_{12|*}\,\sigma_{1|*}\,\sigma_{2|*} & \cdots & \rho_{1n|*}\,\sigma_{1|*}\,\sigma_{n|*} \\[4pt]
   \rho_{21|*}\,\sigma_{2|*}\,\sigma_{1|*} & \sigma_{2|*}^2 & \cdots & \rho_{2n|*}\,\sigma_{2|*}\,\sigma_{n|*} \\[4pt]
   \vdots & \vdots & \ddots & \vdots \\[4pt]
   \rho_{n1|*}\,\sigma_{n|*}\,\sigma_{1|*} & \rho_{n2|*}\,\sigma_{n|*}\,\sigma_{2|*} & \cdots & \sigma_{n|*}^2
   \end{bmatrix}

where :math:`\sigma_{i|*} \equiv \sigma_{\ln IM_i \mid \ln IM^*}` and
:math:`\rho_{ij|*} \equiv \rho_{\ln IM_i, \ln IM_j \mid \ln IM^*}`.

References
----------

- Aristeidou, S., Shahnazaryan, D., O'Reilly, G. J. (2025). Correlation models
  for next-generation amplitude and cumulative intensity measures using
  artificial neural networks. *Earthquake Spectra*, 41(1), 851-875. DOI:
  `10.1177/87552930241270563 <https://doi.org/10.1177/87552930241270563>`_
- Baker, J. W. (2011). Conditional mean spectrum: tool for ground-motion
  selection. *Journal of Structural Engineering*, 137(3), 322-331. DOI:
  `10.1061/(ASCE)ST.1943-541X.0000215 <https://doi.org/10.1061/(ASCE)ST.1943-541X.0000215>`_
- Baker, J. W., Bradley, B. A. (2017). Intensity measure correlations observed
  in the NGA-West2 database, and dependence of correlations on rupture and site
  parameters. *Earthquake Spectra*, 33(1), 145-156. DOI:
  `10.1193/060716eqs095m <https://doi.org/10.1193/060716eqs095m>`_
- Baker, J. W., Lee, C. (2017). An improved algorithm for selecting ground
  motions to match a conditional spectrum. *Journal of Earthquake
  Engineering*, 22(4), 708-723. DOI:
  `10.1080/13632469.2016.1264334 <https://doi.org/10.1080/13632469.2016.1264334>`_
- Bradley, B. A. (2010). A generalized conditional intensity measure approach
  and holistic ground-motion selection. *Earthquake Engineering & Structural
  Dynamics*, 39(12), 1321-1342. DOI:
  `10.1002/eqe.995 <https://doi.org/10.1002/eqe.995>`_
- Bradley, B. A. (2012). A ground motion selection algorithm based on the
  generalized conditional intensity measure approach. *Soil Dynamics and
  Earthquake Engineering*, 40, 48-61. DOI:
  `10.1016/j.soildyn.2012.04.007 <https://doi.org/10.1016/j.soildyn.2012.04.007>`_
- Jayaram, N., Lin, T., Baker, J. W. (2011). A computationally efficient
  ground-motion selection algorithm for matching a target response spectrum
  mean and variance. *Earthquake Spectra*, 27(3), 797-815. DOI:
  `10.1193/1.3608002 <https://doi.org/10.1193/1.3608002>`_
- Lam, S. K., Pitrou, A., Seibert, S. (2015). Numba: a LLVM-based Python JIT
  compiler. *Proceedings of the Second Workshop on the LLVM Compiler
  Infrastructure in HPC*, 1-6. DOI:
  `10.1145/2833157.2833162 <https://doi.org/10.1145/2833157.2833162>`_
