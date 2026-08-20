"""
A generalized conditional intensity measure approach.

Bradley, B. A. (2010). A generalized conditional intensity measure approach and
holistic ground-motion selection. Earthquake Engineering & Structural Dynamics,
39(12), 1321-1342. DOI: 10.1002/eqe.995

Rock site in Christchurch, New Zealand, Vs30 = 760 m/s, conditioned on
Sa(1.0s) = 0.165 g, the 2% in 50 year value. The intensity measure vector is
{SA, SI, ASI, Ia, Ds595}, which is what makes this case worth replicating: it
exercises the period-independent IMs and their correlations, not only spectral
accelerations.

Inputs
------
The published target below is the numerical output of the author's own
implementation, which tabulates each CDF on a grid of lognormal quantiles and
so carries the mixture median and dispersion exactly.

The rupture set, in 'assets/', is the Sa(1.0s) = 0.165 g, 1/2475
disaggregation for Christchurch (172.6E, 43.5S) from the NZ national seismic
hazard model as implemented in OpenSHA, supplied by the author: 1000 scenarios
with magnitude, source-to-site distance, epsilon, rake and percent
contribution. Percent contributions sum to 92.78, the list being truncated to
the largest contributors, and are normalised by that sum, as the author's
implementation does.

Three conventions of the 2010 implementation are followed, which the author
confirmed:

1. a single source-to-site distance per rupture, used as the distance input to
   every ground motion model, so rjb and rrup are both set to it;
2. no ztor and no d_hyp, neither being an input to any model used here;
3. focal mechanism entering as a per-rupture rake. A site level 'mechanism' is
   deliberately not set, since djura derives rake from mechanism only when rake
   is absent and the disaggregation spans three mechanisms.

Epsilon, and why two comparisons are made
-----------------------------------------
The article takes epsilon per rupture from the disaggregation itself, which the
NZ national seismic hazard model produced with its own ground motion model,
while the intensity measures are predicted with Boore and Atkinson (2008).
djura instead derives epsilon from the model it is given for the conditioning
intensity measure, since it carries neither the NZ model nor an input for
supplying epsilon directly. Either is a reasonable choice; they simply differ.

Both comparisons are therefore made. Substituting the published epsilon into
djura's own per-rupture moments leaves only its BA08 predictions and its
combination over the 1000 ruptures, and reproduces the article to a fraction of
a percent. As djura is configured, the medians still reproduce the article.

Component models of the period
------------------------------
The case is from 2010-2012 and several component models have since been
superseded, so exact reproduction needs the models of the time rather than
their replacements. Three such differences remain, all understood:

* ASI. The article used a pre-publication ASI-SA correlation, later published
  as Bradley (2011), which is what djura implements: 0.587 against the 0.61 of
  table I. Not to be 'fixed'.
* Ds595. The article takes rho(Ds595, SA) = 0, no correlation model having
  existed in 2010, and its duration is the Abrahamson and Silva base model
  with the 5-95% dispersion of 0.49 and D_ratio left at zero. djura applies
  both the Bradley (2011) correlation and the D_ratio that converts the base
  5-75% model into 5-95%, which is why its Ds595 median is about 2.2 times the
  published one. Setting the two back to the article's choices reproduces the
  published value exactly.
* SA(10s) is absent from the target. djura's SI-SA and related correlation
  functions guard with '0.01 <= period < 10', excluding the endpoint, although
  their piecewise breakpoints close the last segment at 10s.

Correlation models
------------------
Several of the pairs needed here have more than one model registered in djura,
so the ones the article used are named explicitly through 'correlation-models'.
With those supplied, table I is reproduced.

Notes
-----
The prospective database is the bundled ESM flatfile, which is the only one
carrying observed ASI and SI. The article selected from NGA-West1. This module
validates the target distributions, which are deterministic given the inputs.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import djura.data_loader as data_loader
from djura.record_selection.gcim import GCIM
from djura.record_selection.gsim import models as gsim_models

#: Every test here replicates a published case study, so the whole
#: module carries the marker
pytestmark = pytest.mark.validation

flatfile_dir = Path(
    __file__).resolve().parents[3] / "src/djura/record_selection/assets"

#: The only bundled database carrying observed ASI and SI, which the intensity
#: measure vector of this case requires
#: this is the default flatfile for the record selector
DATABASE = flatfile_dir / "flatfile_shallow_v1.pickle"

#: Disaggregation behind figure 1
DEAGGREGATION = (Path(__file__).resolve().parent
                 / "assets/bradley2010_deagg_sa1p0_2pct50.csv")

#: Conditioning intensity measure, the 2% in 50 year Sa(1.0s) in [g]
IM_STAR_PERIOD, IM_STAR_VALUE = 1.0, 0.165

#: Site conditions. 'soil' is the rock indicator required by
#: AbrahamsonSilva1996, which djura does not derive from vs30. z1pt0 and z2pt5
#: are omitted, no model used here reading them
SITE_PARAMETERS = {"vs30": 760, "soil": 0}

#: Ground motion models of the article. SI and ASI are indirect models,
#: deriving their intensity measure from an arbitrary spectral acceleration
#: model, so they take that model as an argument
GMMS = [
    {"SA": {"names": ["BooreAtkinson2008"], "weights": [1]}},
    {"SI": {"names": ["BradleyEtAl2009SI"], "weights": [1],
            "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
    {"ASI": {"names": ["Bradley2010ASI"], "weights": [1],
             "kwargs": [{"gmpe": "BooreAtkinson2008"}]}},
    {"IA": {"names": ["TravasarouEtAl2003"], "weights": [1]}},
    {"Ds595": {"names": ["AbrahamsonSilva1996"], "weights": [1]}},
]

#: Correlation models of the article, named explicitly. Each of these pairs has
#: more than one model registered in djura, so the choice is supplied rather
#: than left to the default
CORRELATION_MODELS = {
    "SA-SA": "baker_jayaram",        # Baker and Jayaram (2008)
    "IA-SA": "baker2007_ia_sa",      # Baker (2007)
    "Ds595-SA": "bradley2011_ds595_sa",
}

#: Published target for spectral acceleration, {period: (median [g], sigma)},
#: being the CMSdist array. The conditioning period, where the distribution is
#: degenerate, and 10s, which the correlation functions exclude, are omitted
PUBLISHED_SA = {
    0.05: (0.1267, 0.6803),
    0.10: (0.1535, 0.7842),
    0.15: (0.2008, 0.7552),
    0.20: (0.2268, 0.7291),
    0.30: (0.2355, 0.6416),
    0.40: (0.2341, 0.5834),
    0.50: (0.2189, 0.5406),
    1.50: (0.0979, 0.6282),
    2.00: (0.0630, 0.7286),
    2.50: (0.0428, 0.7988),
    3.00: (0.0313, 0.8543),
    4.00: (0.0192, 0.9342),
    5.00: (0.0141, 1.0199),
    7.50: (0.0067, 1.1573),
}

#: Published target for the period-independent IMs, {IM: (median, sigma)}, in
#: cm.s, g.s, m/s and s respectively
PUBLISHED_IMS = {
    "SI": (49.9512, 0.4995),
    "ASI": (0.0914, 0.6017),
    "IA": (0.3179, 0.9955),
    "Ds595": (8.7919, 0.8061),
}

#: Table I, rho(IMi, Sa(1.0)), reported rather than asserted
PUBLISHED_RHO = {"SA(0.05s)": 0.42, "SA(0.5s)": 0.75,
                 "SI": 0.92, "ASI": 0.61, "IA": 0.7}

#: D_ratio applied by djura to convert the Abrahamson and Silva base model,
#: which is Ds5-75, into Ds5-95. The article leaves it at zero
D_RATIO = 0.845

#: Tolerance when the published epsilon is substituted, so that the only
#: remaining differences are the ground motion model predictions and the
#: combination over ruptures
EXACT_TOLERANCE = 0.015

#: Tolerance on the medians as djura is configured, epsilon derived from BA08
MEDIAN_TOLERANCE = 0.015

#: Allowance on the requirement that the dispersion not exceed the published
#: value
DISPERSION_TOLERANCE = 0.05

#: ASI is held looser. Its correlation with Sa(1.0) is the one place djura
#: implements a later model than the article, 0.587 against 0.61, and that
#: carries into the median through the epsilon shift: 2% low, not 1%
ASI_MEDIAN_TOLERANCE = 0.03

PERIODS = sorted(PUBLISHED_SA)

#: The article's intensity measure vector, one panel per entry in the figures
PANEL_IMS = ["SA(0.05s)", "SA(0.5s)", "SI", "ASI", "IA", "Ds595"]

AXIS_LABELS = {"SA(0.05s)": "Sa(0.05s) [g]", "SA(0.5s)": "Sa(0.5s) [g]",
               "SI": "SI [cm.s]", "ASI": "ASI [g.s]", "IA": "Ia [m/s]",
               "Ds595": "Ds595 [s]"}

#: Panels where a component model of the period differs, annotated so the
#: difference reads as documented rather than as a fault
PANEL_NOTES = {"ASI": "later ASI-SA correlation",
               "Ds595": "rho = 0 and no D_ratio in the article"}

#: Causal bands of the two suites of table II. Selection is weighted on the
#: article's intensity measure vector and on nothing else
SUITE_BANDS = {
    "Suite 1": {"magnitude": [4.0, 6.0], "Rjb": [0, 20]},
    "Suite 2": {"magnitude": [7.0, 9.0], "Rjb": [50, 500]},
}

NUM_RECORDS = 15


def read_deaggregation():
    """Rupture scenarios and the published epsilon of the disaggregation

    Returns
    -------
    tuple
        The rupture list for djura, and the per-rupture epsilon of Sa(1.0) as
        the NZ hazard model computed it
    """
    data = np.genfromtxt(DEAGGREGATION, delimiter=",", names=True,
                         dtype=None, encoding="utf8")

    weights = data["percent_contribution"] / data["percent_contribution"].sum()
    ruptures = [
        {"mag": float(mag), "rjb": float(distance), "rrup": float(distance),
         "rake": float(rake), "weight": float(weight)}
        for mag, distance, rake, weight
        in zip(data["magnitude"], data["source_to_site_distance_km"],
               data["rake_deg"], weights)
    ]

    return ruptures, data["epsilon"]


def resolve_gmm(name, **kwargs):
    """Instantiate a ground motion model by name, recursing into 'gmpe'

    The indirect models take the underlying spectral acceleration model as an
    instance, whose REQUIRES_* attributes they copy onto themselves, so a name
    cannot be passed through.
    """
    if isinstance(kwargs.get("gmpe"), str):
        kwargs = dict(kwargs, gmpe=resolve_gmm(kwargs["gmpe"]))

    return getattr(gsim_models, name)(**kwargs)


def selection_weights(imi):
    """Equal weight on the article's vector, zero on everything else"""
    weight = 1.0 / len(PANEL_IMS)

    return [weight if im in PANEL_IMS else 0.0 for im in imi]


def build_input(ruptures: list, context_limits: dict = None) -> dict:
    """Assemble the GCIM input

    Parameters
    ----------
    ruptures : list
        Rupture scenarios of the disaggregation
    context_limits : dict, optional
        Causal parameter band, by default None (no screening, targets only)

    Returns
    -------
    dict
        Input arguments for GCIM creation and selection
    """
    gmms = []
    for entry in GMMS:
        resolved = {}
        for im, spec in entry.items():
            spec = dict(spec)
            if "kwargs" in spec:
                spec["kwargs"] = [
                    {key: resolve_gmm(value) if key == "gmpe" else value
                     for key, value in kwargs.items()}
                    for kwargs in spec["kwargs"]
                ]
            resolved[im] = spec
        gmms.append(resolved)

    imi = [f"SA({period}s)" for period in PERIODS] + list(PUBLISHED_IMS)

    return {
        "gmms": gmms,
        "correlation-models": dict(CORRELATION_MODELS),
        "site-parameters": dict(SITE_PARAMETERS),
        "ruptures": ruptures,
        "imi": imi,
        "im-star": {
            "type": f"SA({IM_STAR_PERIOD})",
            "value": IM_STAR_VALUE,
            "gmms": {"names": ["BooreAtkinson2008"], "weights": [1]},
        },
        # Two components with the geometric mean, the article taking the
        # geometric mean of the two horizontal components and BA08 predicting
        # GMRotI50. With one component djura stacks the components as separate
        # candidates and ignores the component definition
        "num-components": 2,
        "component-definition": "geomean",
        "num_records": NUM_RECORDS,
        "nreplicate": 1,
        "seed": 1,
        "ks_alpha": 0.1,
        "max_scaling_factor": 100,
        "context_limits": dict(context_limits or {}),
        "im_weights": selection_weights(imi),
    }


def combine_over_ruptures(mu, sigma, rho, epsilon, weights):
    """Mixture median and log-dispersion of the conditional distribution

    The article's equations (8) to (12), being the mean and variance of the
    logarithm of a mixture of lognormals over the disaggregated ruptures.

    Parameters
    ----------
    mu, sigma : numpy.ndarray
        Per-rupture unconditional mean and standard deviation of the logarithm
    rho : float
        Correlation of the intensity measure with the conditioning one
    epsilon : numpy.ndarray
        Per-rupture epsilon of the conditioning intensity measure
    weights : numpy.ndarray
        Rupture contributions, summing to one

    Returns
    -------
    tuple
        Median in the units of the intensity measure, and standard deviation of
        the logarithm
    """
    conditional_mu = mu + sigma * rho * epsilon
    mean = (weights * conditional_mu).sum()
    variance = (weights * ((1 - rho ** 2) * sigma ** 2
                           + (conditional_mu - mean) ** 2)).sum()

    return float(np.exp(mean)), float(np.sqrt(variance))


def target_moments(output_create, im, index=0):
    """Median and log-dispersion of one entry of djura's target"""
    median = float(np.exp(np.asarray(
        output_create["target"]["mu_lnIMi"][im]).reshape(-1)[index]))
    sigma = float(np.asarray(
        output_create["target"]["sigma_lnIMi"][im]).reshape(-1)[index])

    return median, sigma


def published_epsilon_moments(output_create, im, rho, epsilon, index=0):
    """Target moments with the published epsilon substituted"""
    data = output_create["data"]

    return combine_over_ruptures(
        data["mu_lnIMi_rup"][im][index],
        data["sigma_lnIMi_rup"][im][index],
        rho,
        epsilon,
        np.asarray(data["weights_imi"][im], float))


def derived_epsilon(output_create):
    """Per-rupture epsilon of the conditioning IM as djura derives it"""
    per_rupture = output_create["data"]["epsilon_lnIMj_rup"]

    return np.array([list(per_rupture[i].values())[0][0]
                     for i in sorted(per_rupture)])


def correlations(output_create):
    """rho(IMi, Sa(1.0)) for every entry of the target, keyed as 'imi' is"""
    rho = output_create["corr_imi_imj"]
    periods = np.asarray(output_create["target"]["IMi"]["SA"], float)

    out = {f"SA({period}s)": float(value) for period, value
           in zip(periods, np.asarray(rho["SA"], float).reshape(-1))}
    for im in PUBLISHED_IMS:
        out[im] = float(np.asarray(rho[im], float).reshape(-1)[0])

    return out


@pytest.fixture(scope="module")
def deaggregation():
    """Rupture scenarios and the published epsilon"""
    return read_deaggregation()


@pytest.fixture(scope="module")
def output_create(deaggregation):
    """The target built once and shared, create() over 1000 ruptures being slow

    The metadata is cached for the lifetime of the process, so the cache is
    invalidated on the way in and out.
    """
    if not DATABASE.exists():
        pytest.skip(f"{DATABASE.name} is not available")

    ruptures, _ = deaggregation

    previous_path = os.environ.get("DJURA_METADATA_PATH")
    previous_metadata = data_loader._metadata

    os.environ["DJURA_METADATA_PATH"] = str(DATABASE)
    data_loader._metadata = None
    try:
        gcim = GCIM(build_input(ruptures), conditional=True)
        gcim.create()
        yield gcim.output_create
    finally:
        if previous_path is None:
            os.environ.pop("DJURA_METADATA_PATH", None)
        else:
            os.environ["DJURA_METADATA_PATH"] = previous_path
        data_loader._metadata = previous_metadata


@pytest.mark.slow
class TestCase1:

    def test_target_matches_the_published_target(self, output_create):
        """The target of figures 2 and 3, as djura is configured

        Medians must reproduce the article. The dispersion is governed by where
        epsilon comes from, so it is required only not to exceed the published
        value, self-consistent epsilon being able to remove between-rupture
        variance but not add it. Ds595 is excluded, standing apart on both the
        correlation and the duration ratio.
        """
        for index, period in enumerate(PERIODS):
            published_median, published_sigma = PUBLISHED_SA[period]
            median, sigma = target_moments(output_create, "SA", index)

            assert median == pytest.approx(
                published_median, rel=MEDIAN_TOLERANCE), f"median at {period}s"
            assert sigma <= published_sigma * (1 + DISPERSION_TOLERANCE), \
                f"sigma at {period}s"

        for im in ("SI", "ASI", "IA"):
            published_median, published_sigma = PUBLISHED_IMS[im]
            median, sigma = target_moments(output_create, im)
            tolerance = (ASI_MEDIAN_TOLERANCE if im == "ASI"
                         else MEDIAN_TOLERANCE)

            assert median == pytest.approx(
                published_median, rel=tolerance), f"{im} median"
            assert sigma <= published_sigma * (
                1 + DISPERSION_TOLERANCE), f"{im} sigma"

    def test_target_is_exact_given_the_published_epsilon(
            self, output_create, deaggregation):
        """The published target is reproduced once epsilon is taken from it

        Substituting the epsilon the article used leaves only djura's ground
        motion model predictions and its combination over the 1000 ruptures,
        so this is the tight check on both. Ds595 is included with the
        article's own two choices restored, rho = 0 and no duration ratio.
        """
        _, epsilon = deaggregation
        rho = correlations(output_create)

        for index, period in enumerate(PERIODS):
            published_median, published_sigma = PUBLISHED_SA[period]
            median, sigma = published_epsilon_moments(
                output_create, "SA", rho[f"SA({period}s)"], epsilon, index)

            assert median == pytest.approx(
                published_median, rel=EXACT_TOLERANCE), f"median at {period}s"
            assert sigma == pytest.approx(
                published_sigma, rel=EXACT_TOLERANCE), f"sigma at {period}s"

        for im in ("SI", "ASI", "IA"):
            published_median, published_sigma = PUBLISHED_IMS[im]
            median, sigma = published_epsilon_moments(
                output_create, im, rho[im], epsilon)

            assert median == pytest.approx(
                published_median, rel=2 * EXACT_TOLERANCE), f"{im} median"
            assert sigma == pytest.approx(
                published_sigma, rel=2 * EXACT_TOLERANCE), f"{im} sigma"

        data = output_create["data"]
        published_median, published_sigma = PUBLISHED_IMS["Ds595"]
        median, sigma = combine_over_ruptures(
            data["mu_lnIMi_rup"]["Ds595"][0] - D_RATIO,
            data["sigma_lnIMi_rup"]["Ds595"][0],
            0.0,
            epsilon,
            np.asarray(data["weights_imi"]["Ds595"], float))

        assert median == pytest.approx(published_median, rel=EXACT_TOLERANCE)
        assert sigma == pytest.approx(published_sigma, rel=EXACT_TOLERANCE)

    def test_report_comparison(self, output_create, deaggregation, report_dir):
        """Tabulate the computed target against the published one

        Everything informative but not worth asserting goes here: the
        correlations of table I, the two epsilon populations, and both variants
        of every moment.
        """
        _, epsilon = deaggregation
        rho = correlations(output_create)
        derived = derived_epsilon(output_create)
        weights = np.asarray(
            output_create["data"]["weights_imi"]["SA"], float)

        rows = [
            "# Bradley (2010), case 1",
            "",
            f"Prospective database: `{DATABASE.name}`.",
            "",
            "'djura' is `create()` as configured, deriving epsilon from Boore "
            "and Atkinson.",
            "'published eps' substitutes the epsilon column of the "
            "disaggregation into djura's",
            "own per-rupture moments, which is what the article did.",
            "",
            "## Correlations with Sa(1.0), table I",
            "",
            "| IM | Published | djura | Difference |",
            "| --- | --- | --- | --- |",
        ]
        for im, published in PUBLISHED_RHO.items():
            rows.append(f"| {im} | {published:.2f} | {rho[im]:.4f} | "
                        f"{rho[im] - published:+.4f} |")
        rows += [
            f"| Ds595 | - (taken as zero) | {rho['Ds595']:.4f} | - |",
            "",
            "ASI is the one pair for which djura implements a later model "
            "than the article,",
            "which used a pre-publication equation. The difference is real "
            "and is not a fault.",
            "",
            "## Epsilon of Sa(1.0)",
            "",
            "| Quantity | Published column | djura |",
            "| --- | --- | --- |",
            f"| Hazard-weighted mean | {(weights * epsilon).sum():.4f} | "
            f"{(weights * derived).sum():.4f} |",
            f"| Standard deviation | {epsilon.std():.4f} | "
            f"{derived.std():.4f} |",
            f"| Correlation between the two | "
            f"{np.corrcoef(derived, epsilon)[0, 1]:.4f} | |",
            f"| Largest difference | "
            f"{np.abs(derived - epsilon).max():.4f} | |",
            "",
            "## Spectral acceleration target",
            "",
            "| T [s] | Published median | djura | published eps | "
            "Published sigma | djura | published eps |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for index, period in enumerate(PERIODS):
            published_median, published_sigma = PUBLISHED_SA[period]
            median, sigma = target_moments(output_create, "SA", index)
            exact_median, exact_sigma = published_epsilon_moments(
                output_create, "SA", rho[f"SA({period}s)"], epsilon, index)
            rows.append(
                f"| {period} | {published_median:.4f} | {median:.4f} | "
                f"{exact_median:.4f} | {published_sigma:.4f} | {sigma:.4f} | "
                f"{exact_sigma:.4f} |")

        rows += [
            "",
            "## Period-independent intensity measures",
            "",
            "| IM | Published median | djura | published eps | "
            "Published sigma | djura | published eps |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for im, (published_median, published_sigma) in PUBLISHED_IMS.items():
            median, sigma = target_moments(output_create, im)
            exact_median, exact_sigma = published_epsilon_moments(
                output_create, im, rho[im], epsilon)
            rows.append(
                f"| {im} | {published_median:.4f} | {median:.4f} | "
                f"{exact_median:.4f} | {published_sigma:.4f} | {sigma:.4f} | "
                f"{exact_sigma:.4f} |")

        rows += [
            "",
            "Ds595 stands apart on both counts: the article takes "
            "rho(Ds595, SA) = 0, and it",
            "leaves the Abrahamson and Silva D_ratio of "
            f"{D_RATIO} at zero, which converts the base",
            "5-75% model into 5-95%. Restoring both reproduces the published "
            "value exactly.",
            "",
        ]

        path = report_dir / "case1_comparison_esm.md"
        path.write_text("\n".join(rows))


def mixture_cdf(output, im, index=0, points=400):
    """Grid and CDF of one target entry, as a mixture over the ruptures

    The target is a mixture of lognormals, not a lognormal, so the curve is
    built from the per-rupture conditional moments rather than from the
    reported median and dispersion.

    Parameters
    ----------
    output : dict
        Output of GCIM.create()
    im : str
        Intensity measure key of the target
    index : int, optional
        Entry within that key, by default 0
    points : int, optional
        Number of grid points, by default 400

    Returns
    -------
    tuple
        Intensity measure values and the cumulative probabilities at them
    """
    from scipy.stats import norm

    data = output["data"]
    mu = data["mu_lnIMi_lnIMj_rup"][im][index]
    sigma = data["sigma_lnIMi_lnIMj_rup"][im][index]
    weights = np.asarray(data["weights_imi"][im], float)

    grid = np.linspace((mu - 4 * sigma).min(), (mu + 4 * sigma).max(), points)
    cdf = (weights[:, None]
           * norm.cdf((grid[None, :] - mu[:, None]) / sigma[:, None])).sum(0)

    return np.exp(grid), cdf / weights.sum()


def panel_mixture(output, im):
    """Grid and CDF for one figure panel, spectral entries found by period"""
    if im.startswith("SA"):
        return mixture_cdf(output, "SA", PERIODS.index(float(im[3:-2])))

    return mixture_cdf(output, im)


def panel_grid(plt, title):
    """A two by three grid of axes, one per intensity measure"""
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    figure.suptitle(title)

    return figure, dict(zip(PANEL_IMS, axes.ravel()))


def save_panels(figure, axes, path):
    """Label, annotate and write a panel figure"""
    handles, labels = list(axes.values())[0].get_legend_handles_labels()
    for im, axis in axes.items():
        axis.set_xscale("log")
        axis.set_xlabel(AXIS_LABELS[im])
        axis.set_ylabel("Cumulative probability")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.3)
        if im in PANEL_NOTES:
            axis.set_title(PANEL_NOTES[im], fontsize=9, color="0.35")

    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(path, dpi=130)


def published_median(im):
    """Published median of one panel intensity measure"""
    if im.startswith("SA"):
        return PUBLISHED_SA[float(im[3:-2])][0]

    return PUBLISHED_IMS[im][0]


@pytest.fixture(scope="module")
def pyplot():
    """pyplot with a non-interactive backend, skipping when absent"""
    matplotlib = pytest.importorskip(
        "matplotlib", reason="install djura[plot] to redraw the figures")
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


@pytest.mark.slow
class TestCase1Figures:

    def test_plot_target_spectrum(self, output_create, pyplot, plot_dir):
        """Figure 2, the conditional distribution of Sa"""
        periods = np.asarray(output_create["target"]["IMi"]["SA"], float)
        mu = np.asarray(
            output_create["target"]["mu_lnIMi"]["SA"], float).reshape(-1)
        sigma = np.asarray(
            output_create["target"]["sigma_lnIMi"]["SA"], float).reshape(-1)

        # The conditioning period is not an entry of the target, the
        # distribution being degenerate there, so it is inserted for the
        # drawing: the curve passes through the conditioning amplitude with no
        # dispersion
        order = np.argsort(np.append(periods, IM_STAR_PERIOD))
        drawn = np.append(periods, IM_STAR_PERIOD)[order]
        median = np.append(np.exp(mu), IM_STAR_VALUE)[order]
        dispersion = np.append(sigma, 0.0)[order]

        published = np.array([(period, *PUBLISHED_SA[period])
                              for period in PERIODS])

        figure, axis = pyplot.subplots(figsize=(7.5, 5.5))
        axis.plot(drawn, median, color="C0", lw=2.0, label="djura, median")
        axis.plot(drawn, median * np.exp(-dispersion), color="C0", lw=1.0,
                  ls=":", label="djura, 16th and 84th")
        axis.plot(drawn, median * np.exp(dispersion), color="C0", lw=1.0,
                  ls=":")
        axis.plot(published[:, 0], published[:, 1], color="k", ls="--",
                  lw=1.6, label="article, median")
        axis.plot(published[:, 0], published[:, 1] * np.exp(-published[:, 2]),
                  color="0.4", lw=1.0, ls=":",
                  label="article, 16th and 84th")
        axis.plot(published[:, 0], published[:, 1] * np.exp(published[:, 2]),
                  color="0.4", lw=1.0, ls=":")
        axis.plot(IM_STAR_PERIOD, IM_STAR_VALUE, "o", color="C3", ms=7,
                  label=f"Sa({IM_STAR_PERIOD}) = {IM_STAR_VALUE} g")

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Period [s]")
        axis.set_ylabel("Spectral acceleration [g]")
        axis.set_title("Figure 2, conditional distribution of Sa given "
                       f"Sa({IM_STAR_PERIOD}) = {IM_STAR_VALUE} g")
        axis.grid(which="both", alpha=0.3)
        axis.legend(fontsize=9)
        figure.tight_layout()
        figure.savefig(plot_dir / "case1_figure2.png", dpi=130)

    def test_plot_target_distributions(self, output_create, pyplot, plot_dir):
        """Figure 3, the conditional distribution of each intensity measure"""
        figure, axes = panel_grid(
            pyplot, "Figure 3, conditional distributions given "
                    f"Sa({IM_STAR_PERIOD}) = {IM_STAR_VALUE} g")

        for im, axis in axes.items():
            grid, cdf = panel_mixture(output_create, im)
            axis.plot(grid, cdf, color="C0", lw=2.0, label="djura")
            axis.axvline(published_median(im), color="k", ls="--", lw=1.6,
                         label="article, median")

        save_panels(figure, axes, plot_dir / "case1_figure3.png")

    @pytest.mark.parametrize("suite, number", [("Suite 1", 4), ("Suite 2", 5)])
    def test_plot_selected_suite(self, deaggregation, output_create, suite,
                                 number, pyplot, plot_dir):
        """Figures 4 and 5, a selected suite against the target

        The suite is selected within the causal band of the published suite of
        the same number, from ESM rather than from the NGA-West1 database of
        the article, so the records differ and it is the distribution that is
        compared. The Kolmogorov-Smirnov bounds are djura's own.
        """
        from scipy.stats import kstwobign

        ruptures, _ = deaggregation
        gcim = GCIM(build_input(ruptures, SUITE_BANDS[suite]),
                    conditional=True)
        gcim.create()
        gcim.select()

        records = gcim.records["selected_scaled_best"]
        scaled = np.asarray(records["Scaled_IMs"], float)
        imi = [f"SA({period}s)" for period in PERIODS] + list(PUBLISHED_IMS)
        band = SUITE_BANDS[suite]

        figure, axes = panel_grid(
            pyplot, f"Figure {number}, {suite.lower()}, magnitude "
                    f"{band['magnitude'][0]} to {band['magnitude'][1]} and "
                    f"Rjb {band['Rjb'][0]} to {band['Rjb'][1]} km")

        width = kstwobign.ppf(1 - records["ks_alpha"]) / np.sqrt(len(scaled))
        for im, axis in axes.items():
            grid, cdf = panel_mixture(output_create, im)
            axis.plot(grid, cdf, color="C0", lw=2.0, label="djura target")
            axis.plot(grid, np.clip(cdf - width, 0, 1), color="C0", lw=0.7,
                      ls="-.", label="Kolmogorov-Smirnov bounds")
            axis.plot(grid, np.clip(cdf + width, 0, 1), color="C0", lw=0.7,
                      ls="-.")

            values = np.sort(scaled[:, imi.index(im)])
            axis.step(values, np.arange(1, len(values) + 1) / len(values),
                      where="post", color="C2", lw=1.4,
                      label="selected suite, ESM")
            axis.axvline(published_median(im), color="k", ls="--", lw=1.6,
                         label="article target, median")

        save_panels(figure, axes, plot_dir / f"case1_figure{number}.png")
