"""
A ground motion selection algorithm based on the GCIM approach.

Bradley, B. A. (2012). A ground motion selection algorithm based on the
generalized conditional intensity measure approach. Soil Dynamics and
Earthquake Engineering, 40, 48-61. DOI: 10.1016/j.soildyn.2012.04.007

Site in Los Angeles, Vs30 = 760 m/s, conditioned on SA(3.0s) at three
exceedance levels. The result of the article is the effect of the weight
vector, section 4.2: weighting spectral acceleration alone leaves the CAV and
duration distributions of the selected suite biased, and moving thirty per cent
of the weight onto CAV and the two durations removes the bias while matching
the spectral ordinates just as well. That contrast, and the target it is
measured against, are what this module checks.

Inputs
------
The published targets below are the numerical output of the author's own
implementation, which tabulates each distribution on a grid of lognormal
quantiles and so carries the median exactly.

The rupture sets are the asset ``bradley2012_deagg_sa3pt0.json``. No
rupture-by-rupture listing was archived, the probabilities having been computed
internally, so the bars of the published disaggregation stand in for it: 62
weighted magnitude-distance cells per level, taken at bin centres of 10 km and
0.5 magnitude units.

Why the target is not reproduced exactly
----------------------------------------
Two consequences of working from the binned disaggregation, both bounded and
both understood.

The plotted disaggregation stops at 110 km, so the recovered bars carry 96.6,
99.2 and 99.8 per cent of the hazard at the 50, 10 and 1 per cent levels. The
missing part is the distant tail, and dropping it raises the medians and
narrows the distribution. The error therefore tracks the truncation: the
dispersion of PGV comes out at 0.55 of the published value at the 50 per cent
level, 0.86 at 10 per cent and 0.92 at 1 per cent. The 50 per cent level is the
weakest of the three and is held to a looser tolerance for that reason.

Collapsing each cell onto its centre removes the spread within it, which
accounts for a residual five to seven per cent dispersion deficit at the 1 per
cent level, where truncation is negligible. It touches only the
between-rupture term, so the measures most correlated with SA(3.0) are
unaffected: DSI, SA(2.0s) and SA(5.0s) all reproduce their dispersion to within
half a per cent.

Only the medians are asserted for that reason; the dispersions are reported.

Known deviations
----------------
* **SA(10.0s)** is dropped from the article's vector of 17, so one published
  column is not checked. Every correlation of a period-independent measure with
  SA guards with ``0.01 <= period < 10``, excluding the endpoint. The weights
  are renormalised over the eight remaining ordinates, which preserves the
  seventy-thirty split between the spectral and the other measures that section
  4.2 is about.
* **Duration** runs six to seventeen per cent below the published median at
  every level. ``BommerEtAl2009RSD`` reads ztor, which the article does not
  state and which is assumed at 3 km here.
* Reverse faulting and dip are likewise assumed, neither being stated. dip is
  inert: with rjb equal to rrup and ztor at least 1, the hanging-wall term of
  CampbellBozorgnia2008, the only place it enters, is zero.

Notes
-----
The prospective database is the bundled dataset; the article used NGA-West1.
The selected records are therefore not the article's, and neither are the
causal parameters of the suite: ESM holds few large-magnitude records, so the
algorithm reaches for smaller ones and scales harder, giving mean magnitudes
0.2 to 0.4 units below the published suites and wider distance dispersions.
What survives that difference, and is asserted, is the contrast between the two
weight vectors.
"""

import json
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

#: The bundled database, the default of the record selector
DATABASE = (Path(__file__).resolve().parents[3]
            / "src/djura/record_selection/assets/flatfile_shallow_v1.pickle")

DEAGGREGATION = (Path(__file__).resolve().parent
                 / "assets/bradley2012_deagg_sa3pt0.json")

#: Spectral periods of the article's intensity measure vector, less 10 s
SA_PERIODS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

#: The period-independent measures, in the order the article lists them
OTHER_IMS = ["PGA", "PGV", "ASI", "SI", "DSI", "CAV", "Ds575", "Ds595"]

IMI = [f"SA({period}s)" for period in SA_PERIODS] + OTHER_IMS

#: Site conditions of section 4. Reverse faulting, ztor and dip are assumed,
#: none being stated by the article
SITE_PARAMETERS = {"vs30": 760, "z2pt5": 1.0, "ztor": 3, "dip": 45,
                   "mechanism": "reverse fault"}

#: Ground motion models of the article. ASI, SI and DSI are indirect models,
#: deriving their measure from a spectral acceleration model, so they take that
#: model as an argument
GMMS = [
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
    {"Ds595": {"names": ["BommerEtAl2009RSD"], "weights": [1]}},
]

#: Correlation models of the article, named explicitly. Each of these pairs has
#: more than one model registered in djura, so the choice is supplied rather
#: than left to the default
CORRELATION_MODELS = {
    "SA-SA": "baker_jayaram",
    "Ds575-SA": "bradley2011_ds575_sa",
    "Ds595-SA": "bradley2011_ds595_sa",
}

#: Published median of every intensity measure at each exceedance level, in g,
#: cm/s, g.s, cm.s, cm.s, g.s, s and s respectively
PUBLISHED_TARGET = {
    "50% in 50 yr": {
        "im_star": 0.03749344,
        "median": {
            "SA(0.05s)": 0.11095, "SA(0.1s)": 0.14492, "SA(0.2s)": 0.20237,
            "SA(0.3s)": 0.19195, "SA(0.5s)": 0.15496, "SA(1.0s)": 0.10021,
            "SA(2.0s)": 0.05759, "SA(5.0s)": 0.01698,
            "PGA": 0.10197, "PGV": 10.96707, "ASI": 0.07841, "SI": 39.4671,
            "DSI": 26.51867, "CAV": 0.35496, "Ds575": 8.30144,
            "Ds595": 15.5906,
        },
    },
    "10% in 50 yr": {
        "im_star": 0.08270873,
        "median": {
            "SA(0.05s)": 0.17551, "SA(0.1s)": 0.22249, "SA(0.2s)": 0.30623,
            "SA(0.3s)": 0.29361, "SA(0.5s)": 0.24772, "SA(1.0s)": 0.17613,
            "SA(2.0s)": 0.11638, "SA(5.0s)": 0.03491,
            "PGA": 0.16824, "PGV": 21.19929, "ASI": 0.12427, "SI": 73.88987,
            "DSI": 55.99241, "CAV": 0.56765, "Ds575": 7.75347,
            "Ds595": 14.96712,
        },
    },
    "1% in 50 yr": {
        "im_star": 0.16783433,
        "median": {
            "SA(0.05s)": 0.23047, "SA(0.1s)": 0.28087, "SA(0.2s)": 0.39517,
            "SA(0.3s)": 0.39183, "SA(0.5s)": 0.35169, "SA(1.0s)": 0.28074,
            "SA(2.0s)": 0.21699, "SA(5.0s)": 0.06529,
            "PGA": 0.23296, "PGV": 35.71227, "ASI": 0.17094, "SI": 126.14805,
            "DSI": 108.09566, "CAV": 0.78779, "Ds575": 7.39456,
            "Ds595": 14.72006,
        },
    },
}

#: Tolerance on the target medians, per level. The 50 per cent level is loosest
#: because its recovered disaggregation is the most truncated, carrying 96.6
#: per cent of the hazard against 99.2 and 99.8
MEDIAN_TOLERANCE = {"50% in 50 yr": 0.35, "10% in 50 yr": 0.15,
                    "1% in 50 yr": 0.15}

#: The measures the two weight vectors disagree about: they carry weight under
#: equation (16) and none under equation (15)
CONTRASTED = ["CAV", "Ds575", "Ds595"]

#: Median amplitude scale factor of the two suites, table 2
PUBLISHED_SCALE_FACTOR = {"sa_only": 1.9, "sa_cav_ds": 1.1}

NUM_RECORDS = 30
MAX_SCALING_FACTOR = 10


def read_ruptures(level):
    """Rupture scenarios of one exceedance level"""
    data = json.loads(DEAGGREGATION.read_text())["levels"][level]

    return [{"mag": mag, "rjb": rjb, "rrup": rrup, "weight": weight}
            for mag, rjb, rrup, weight in data["ruptures"]]


def resolve_gmm(name, **kwargs):
    """Instantiate a ground motion model by name, recursing into 'gmpe'

    The indirect models take the underlying spectral acceleration model as an
    instance, whose REQUIRES_* attributes they copy onto themselves, so a name
    cannot be passed through.
    """
    if isinstance(kwargs.get("gmpe"), str):
        kwargs = dict(kwargs, gmpe=resolve_gmm(kwargs["gmpe"]))

    return getattr(gsim_models, name)(**kwargs)


def weights(vector):
    """Weight per entry of `imi` for one of the article's two vectors

    Equation (15) puts the whole weight on the spectral ordinates in equal
    parts; equation (16) puts seventy per cent there and ten per cent on each
    of CAV, Ds575 and Ds595. Both are renormalised over the eight ordinates
    retained, so the split between the groups is the article's.
    """
    share = {"sa_only": 1.0, "sa_cav_ds": 0.7}[vector]
    others = {im: 0.1 for im in CONTRASTED} if vector == "sa_cav_ds" else {}

    return [share / len(SA_PERIODS) if im.startswith("SA(")
            else others.get(im, 0.0) for im in IMI]


def build_input(level: str, vector: str) -> dict:
    """Assemble the GCIM input for one exceedance level and weight vector

    Parameters
    ----------
    level : str
        Exceedance level, a key of PUBLISHED_TARGET
    vector : str
        'sa_only' for equation (15) or 'sa_cav_ds' for equation (16)

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

    return {
        "gmms": gmms,
        "correlation-models": dict(CORRELATION_MODELS),
        "site-parameters": dict(SITE_PARAMETERS),
        "ruptures": read_ruptures(level),
        "imi": IMI,
        "im-star": {
            "type": "SA(3.0)",
            "value": PUBLISHED_TARGET[level]["im_star"],
            "gmms": {"names": ["BooreAtkinson2008"], "weights": [1]},
        },
        # Two components with the geometric mean, the article working with the
        # geometric mean and BA08 predicting GMRotI50
        "num-components": 2,
        "component-definition": "geomean",
        "num_records": NUM_RECORDS,
        "nreplicate": 1,
        "seed": 1,
        "ks_alpha": 0.1,
        "max_scaling_factor": MAX_SCALING_FACTOR,
        # No causal screening, which is explicit in the article and central to
        # its argument
        "context_limits": {},
        "im_weights": weights(vector),
    }


def target_median(output, im):
    """Median of one entry of djura's target"""
    key = "SA" if im.startswith("SA(") else im
    index = SA_PERIODS.index(float(im[3:-2])) if im.startswith("SA(") else 0

    return float(np.exp(np.asarray(
        output["target"]["mu_lnIMi"][key]).reshape(-1)[index]))


def target_dispersion(output, im):
    """Log-standard deviation of one entry of djura's target"""
    key = "SA" if im.startswith("SA(") else im
    index = SA_PERIODS.index(float(im[3:-2])) if im.startswith("SA(") else 0

    return float(np.asarray(
        output["target"]["sigma_lnIMi"][key]).reshape(-1)[index])


def suite_median(records, im):
    """Median of one intensity measure over the selected suite"""
    scaled = np.asarray(records["selected_scaled_best"]["Scaled_IMs"], float)

    return float(np.median(scaled[:, IMI.index(im)]))


@pytest.fixture(scope="module")
def bundled_database():
    """Run against the bundled database

    Any DJURA_METADATA_PATH already set would substitute another database, so
    the bundled one is named explicitly for the duration and whatever was there
    is restored afterwards. The metadata is cached for the lifetime of the
    process, so that cache is invalidated on the way in and out.
    """
    if not DATABASE.exists():
        pytest.skip(f"{DATABASE.name} is not available")

    previous_path = os.environ.get("DJURA_METADATA_PATH")
    previous_metadata = data_loader._metadata

    os.environ["DJURA_METADATA_PATH"] = str(DATABASE)
    data_loader._metadata = None
    try:
        yield
    finally:
        if previous_path is None:
            os.environ.pop("DJURA_METADATA_PATH", None)
        else:
            os.environ["DJURA_METADATA_PATH"] = previous_path
        data_loader._metadata = previous_metadata


@pytest.fixture(scope="module")
def targets(bundled_database):
    """The target of every exceedance level, built once and shared"""
    outputs = {}
    for level in PUBLISHED_TARGET:
        gcim = GCIM(build_input(level, "sa_cav_ds"), conditional=True)
        gcim.create()
        outputs[level] = gcim.output_create

    return outputs


@pytest.fixture(scope="module")
def suites(bundled_database):
    """The two suites of section 4.2, selected at the 10 per cent level"""
    out = {}
    for vector in ("sa_only", "sa_cav_ds"):
        gcim = GCIM(build_input("10% in 50 yr", vector), conditional=True)
        gcim.create()
        gcim.select()
        out[vector] = gcim.records

    return out


@pytest.mark.slow
class TestCase2:

    def test_target_matches_the_published_target(self, targets):
        """The published median of every measure at every exceedance level"""
        for level, published in PUBLISHED_TARGET.items():
            for im, median in published["median"].items():
                assert target_median(targets[level], im) == pytest.approx(
                    median, rel=MEDIAN_TOLERANCE[level]), f"{im} at {level}"

    def test_weighting_spectral_acceleration_alone_biases_the_others(
            self, suites, targets):
        """Section 4.2, the result of the article

        Under equation (15) the CAV and duration medians of the suite sit well
        above the target, none of them being weighted; under equation (16) they
        come back to it. The spectral ordinates are matched either way,
        which is what makes the comparison a fair one.
        """
        target = targets["10% in 50 yr"]

        for im in CONTRASTED:
            expected = target_median(target, im)
            biased = suite_median(suites["sa_only"], im) / expected
            corrected = suite_median(suites["sa_cav_ds"], im) / expected

            assert biased > 1.5, f"{im} is not biased under equation (15)"
            assert abs(corrected - 1) < abs(biased - 1) / 2, \
                f"{im} is not improved by equation (16)"

        for im in ("SA(0.2s)", "SA(2.0s)"):
            expected = target_median(target, im)
            for vector in ("sa_only", "sa_cav_ds"):
                assert suite_median(suites[vector], im) == pytest.approx(
                    expected, rel=0.2), f"{im} under {vector}"

    def test_equation_16_needs_less_scaling(self, suites):
        """Table 2, the second finding: the fuller vector scales less

        The article reports a median amplitude scale factor of 1.9 against 1.1.
        The levels depend on the database, the ordering does not.
        """
        factors = {
            vector: float(np.median(np.asarray(
                records["selected_scaled_best"]["sf"], float)))
            for vector, records in suites.items()
        }
        largest = max(
            float(np.asarray(records["selected_scaled_best"]["sf"],
                             float).max())
            for records in suites.values())

        assert factors["sa_only"] > factors["sa_cav_ds"]
        assert largest < MAX_SCALING_FACTOR, \
            "the scale factor limit binds, which voids the comparison"

    def test_report_comparison(self, targets, suites, report_dir):
        """Tabulate the target and the suites against the published ones"""
        rows = [
            "# Bradley (2012), case 2",
            "",
            f"Prospective database: `{DATABASE.name}`.",
            "",
            "## Target against the published one",
            "",
            "| IM | " + " | ".join(
                f"{level} median | ratio | dispersion ratio"
                for level in PUBLISHED_TARGET) + " |",
            "| --- " * (1 + 3 * len(PUBLISHED_TARGET)) + "|",
        ]
        reference_dispersion = {
            "50% in 50 yr": {"PGV": 0.7763, "CAV": 0.7505, "PGA": 0.9307,
                             "DSI": 0.2200, "SA(2.0s)": 0.3810},
            "10% in 50 yr": {"PGV": 0.4723, "CAV": 0.4717, "PGA": 0.7016,
                             "DSI": 0.2183, "SA(2.0s)": 0.3885},
            "1% in 50 yr": {"PGV": 0.4283, "CAV": 0.4107, "PGA": 0.6582,
                            "DSI": 0.2174, "SA(2.0s)": 0.3950},
        }
        for im in IMI:
            row = f"| {im} "
            for level in PUBLISHED_TARGET:
                published = PUBLISHED_TARGET[level]["median"][im]
                median = target_median(targets[level], im)
                dispersion = reference_dispersion[level].get(im)
                computed = target_dispersion(targets[level], im)
                ratio = (f"{computed / dispersion:.3f}"
                         if dispersion else "-")
                row += (f"| {published:.4g} | {median / published:.3f} "
                        f"| {ratio} ")
            rows.append(row + "|")

        rows += [
            "",
            "## Section 4.2, median of the suite over the target median",
            "",
            "| IM | equation (15) | equation (16) |",
            "| --- | --- | --- |",
        ]
        target = targets["10% in 50 yr"]
        for im in CONTRASTED + ["PGA", "PGV", "SA(0.2s)", "SA(2.0s)"]:
            expected = target_median(target, im)
            rows.append(
                f"| {im} "
                f"| {suite_median(suites['sa_only'], im) / expected:.3f} "
                f"| {suite_median(suites['sa_cav_ds'], im) / expected:.3f} |")

        rows += [
            "",
            "## Amplitude scale factor, table 2",
            "",
            "| Vector | Published median | djura median |",
            "| --- | --- | --- |",
        ]
        for vector, published in PUBLISHED_SCALE_FACTOR.items():
            factors = np.asarray(
                suites[vector]["selected_scaled_best"]["sf"], float)
            rows.append(f"| {vector} | {published} "
                        f"| {np.median(factors):.2f} |")

        (report_dir / "case2_comparison.md").write_text("\n".join(rows) + "\n")
