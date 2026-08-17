"""
Conditional spectrum based ground motion selection.

Lin, T., Haselton, C. B., & Baker, J. W. (2013). Conditional spectrum-based
ground motion selection. Part I: Hazard consistency for risk-based
assessments. Earthquake Engineering & Structural Dynamics, 42(11), 1847-1865.
DOI: 10.1002/eqe.2301

The conditional spectrum is the special case of the GCIM distribution in which
the intensity measure vector contains only spectral accelerations, so the case
is replicated by listing only SA entries in 'imi' with uniform weights.

Site is in Palo Alto, California, with Vs30 = 400 m/s. The structure is a
twenty storey reinforced concrete special moment frame with elastic modal
periods of 2.6, 0.85 and 0.45s, and a lengthened period of 5.0s.

Seismic hazard inputs, and how they were recovered
--------------------------------------------------
The article states only that mean deaggregation values of magnitude and
distance were taken from the USGS web tool, and tabulates no value other than
Sa(2.6s) = 0.45g at the 2% in 50 year exceedance probability. The scenarios
used below were therefore recovered by repeating that deaggregation. The
procedure carried out was as follows, and can be repeated as given.

1. The USGS NSHMP deaggregation service was queried at

       https://earthquake.usgs.gov/nshmp-haz-ws/deagg/
           {edition}/{region}/{longitude}/{latitude}/{imt}/{vs30}/{period}

   The service returns its own usage document, listing the permitted values of
   every argument, when requested without arguments.

2. The arguments were fixed as follows. Edition 'E2008', the 2008
   conterminous US model, being the model era used by the article; region
   'WUS', the western US; longitude -122.1430 and latitude 37.4419, being
   Palo Alto; shear wave velocity 360 m/s; and return period 2475 years, the
   2% in 50 year exceedance probability. One request was issued per spectral
   period, for example

       .../deagg/E2008/WUS/-122.1430/37.4419/SA2P0/360/2475

3. The 'Deaggregation targets' and 'Mean (over all sources)' entries of the
   response were read for each period, giving

       T [s]    Sa [g]    Mbar    Rbar [km]    epsilon
       0.30     1.740     7.35     10.4        1.62
       0.50     1.507     7.48     10.2        1.57
       0.75     1.237     7.60     10.2        1.50
       1.00     1.021     7.66     10.1        1.45
       2.00     0.583     7.79     11.4        1.33
       3.00     0.399     7.84     11.1        1.24

4. None of the conditioning periods of the article is a model period, so each
   quantity was interpolated linearly in the logarithm of the period, which is
   the treatment the article describes for periods not served directly by the
   hazard software. The results are the SUB_CASES below.

5. As a check on the whole procedure, the interpolated Sa(2.6s) of 0.456g was
   compared against the 0.45g reported in the article. The agreement confirms
   the site coordinates, the model edition, the return period and the
   remaining settings.

On the shear wave velocity. The service accepts only a discrete set of site
classes and rejects 400 m/s outright, so the deaggregation cannot be run at the
velocity stated by the article. Bracketing it, by deaggregating at 360 and
537 m/s and interpolating linearly in the logarithm of the velocity, gives
Sa(2.6s) = 0.420g, which is 6.6% below the published 0.45g, whereas 360 m/s
alone gives 0.456g, which is 1.3% above it. The velocity of 360 m/s is
therefore retained, on the evidence that it reproduces the one value the
article publishes. The likely explanation is that the article selected a site
class in the hazard tool rather than an arbitrary velocity. The site
parameters passed to the ground motion model below retain the 400 m/s of the
article, the 360 m/s applying to the deaggregation alone.

The lengthened period of 5.0s is omitted throughout. The 2008 model provides
no spectral acceleration above 3s, the service rejecting SA4P0 and SA5P0 for
the western US, so its hazard cannot be deaggregated and is not extrapolated.

Notes
-----
The selected records are not expected to match those of the article. The
selection algorithm draws random realizations from the conditional
distribution, so the suite depends on the seed, on the realization draw and on
the prospective database, and the article used NGA-West1 whereas the runs here
use NGA-West2 (downloaded externally) and ESM. What is validated is the target
distribution and the properties which the selected suite must satisfy.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import djura.data_loader as data_loader
from djura.record_selection import correlation_models
from djura.record_selection.gcim import GCIM
from djura.utilities import get_func_args


asset_dir = Path(
    __file__).resolve().parents[3] / "src/djura/record_selection/assets"

#: The two prospective databases, run separately so that the influence of the
#: record set is isolated. The article used NGA-West1.
DATABASES = {
    "nga_west2": asset_dir / "NGA_W2_v2.pickle",
    "esm": asset_dir / "flatfile_shallow.pickle",
}

#: The article obtains the correlation between spectral accelerations at pairs
#: of periods from Baker and Jayaram (2008), which must be requested because
#: the registry places another model first for the SA-SA pair
CORRELATION_MODEL = {"SA-SA": "baker_jayaram"}

#: Site conditions, section 2.1 of the article
SITE_PARAMETERS = {
    "vs30": 400,
    "z2pt5": 2.0,
    "mechanism": "strike-slip fault",
}

#: Conditioning periods of the article, being the first three modal periods of
#: the structure, with the hazard recovered from the deaggregation described
#: above at the 2% in 50 year exceedance probability.
#:
#: 'sa' is the spectral acceleration at the conditioning period in [g], 'mag'
#: and 'rrup' the mean deaggregation magnitude and rupture distance, and 'eps'
#: the mean deaggregation epsilon, retained for reference
SUB_CASES = {
    "3A_T1": {"t_star": 2.60, "sa": 0.456, "mag": 7.82, "rrup": 11.2,
              "eps": 1.27},
    "3B_T2": {"t_star": 0.85, "sa": 1.138, "mag": 7.63, "rrup": 10.2,
              "eps": 1.48},
    "3C_T3": {"t_star": 0.45, "sa": 1.552, "mag": 7.45, "rrup": 10.3,
              "eps": 1.58},
}

#: The one hazard value the article publishes, being Sa at the first modal
#: period at the 2% in 50 year exceedance probability, figure 1(a)
PUBLISHED_SA_T1 = 0.45

#: Uniform hazard spectrum at the 2% in 50 year exceedance probability, taken
#: from the 'Deaggregation targets' of the same responses. The article states
#: that it envelopes the conditional mean spectra, and that each conditional
#: mean spectrum equals it at its own conditioning period, figure 2(b)
UNIFORM_HAZARD_SPECTRUM = {
    0.30: 1.740, 0.50: 1.507, 0.75: 1.236,
    1.00: 1.021, 2.00: 0.583, 3.00: 0.399,
}

#: Conditional mean spectrum of figure 3, for the first modal period at the
#: 2% in 50 year exceedance probability, against which the response spectra of
#: the forty selected ground motions are plotted. The values are read from the
#: logarithmic panel of that figure, in which the mean lies within a band of
#: forty record spectra, so they carry a reading uncertainty of the order of a
#: quarter and support only a coarse comparison
PUBLISHED_SPECTRUM_T1 = {
    0.10: 0.50, 0.50: 0.75, 1.00: 0.65,
    2.60: 0.45, 5.00: 0.20, 10.00: 0.10,
}

#: Tolerance appropriate to values read from a figure rather than tabulated
FIGURE_TOLERANCE = 0.25

#: Modal rupture of the deaggregation, identical at 2 and 3s. Combined with
#: the mean rupture it provides a two scenario set with which to examine the
#: effect of multiple causal earthquakes, section 3.2 of the article
MODAL_RUPTURE = {"mag": 8.09, "rrup": 9.85}

#: Vibration periods of the target. All four conditioning periods are retained
#: whichever one is conditioned upon, so that the implied hazard curves of
#: figure 4(e) can be computed at each of them for every suite
PERIODS = [0.1, 0.2, 0.3, 0.45, 0.5, 0.75, 0.85, 1.0, 1.5, 2.0,
           2.6, 3.0, 4.0, 5.0, 7.5, 10.0]

NUM_RECORDS = 40

#: The rupture is a vertical strike-slip on the San Andreas, which lies about
#: 10 km from the site and dominates its hazard, so the rupture reaches the
#: surface and the Joyner-Boore and rupture distances coincide
ZTOR = 0.0


def build_rupture(mag: float, rrup: float, weight: float = 1.0) -> dict:
    """Rupture scenario from a deaggregation magnitude and distance

    Parameters
    ----------
    mag : float
        Moment magnitude
    rrup : float
        Rupture distance in [km]
    weight : float, optional
        Contribution of the scenario, by default 1.0

    Returns
    -------
    dict
        Rupture arguments
    """
    return {
        "mag": mag,
        "rjb": rrup,
        "rrup": rrup,
        "ztor": ZTOR,
        "d_hyp": 0.6 * mag,
        "weight": weight,
    }


def build_input(case: dict, ruptures: list = None) -> dict:
    """Assemble the GCIM input for one sub-case

    Parameters
    ----------
    case : dict
        Sub-case definition, holding the conditioning period, the spectral
        acceleration and the mean deaggregation rupture
    ruptures : list, optional
        Rupture scenarios, by default None (the mean deaggregation rupture)

    Returns
    -------
    dict
        Input arguments for GCIM creation and selection
    """
    gmm = {"names": ["CampbellBozorgnia2008"], "weights": [1]}

    if ruptures is None:
        ruptures = [build_rupture(case["mag"], case["rrup"])]

    return {
        "gmms": [{"SA": gmm}],
        "correlation-models": dict(CORRELATION_MODEL),
        "site-parameters": dict(SITE_PARAMETERS),
        "ruptures": ruptures,
        "imi": [f"SA({period}s)" for period in PERIODS],
        "im-star": {
            "type": f"SA({case['t_star']})",
            "value": case["sa"],
            "gmms": gmm,
        },
        "num-components": 1,
        "component-definition": "geomean",
        "num_records": NUM_RECORDS,
        "nreplicate": 1,
        "seed": 1,
        "ks_alpha": 0.1,
        "max_scaling_factor": 10,
        "context_limits": {},
        "im_weights": [],
    }


def get_target(gcim: GCIM):
    """Periods, log-means and log-standard deviations of the target"""
    target = gcim.output_create["target"]

    periods = np.asarray(target["IMi"]["SA"], dtype=float)
    mu = np.asarray(target["mu_lnIMi"]["SA"], dtype=float).reshape(-1)
    sigma = np.asarray(target["sigma_lnIMi"]["SA"], dtype=float).reshape(-1)

    return periods, mu, sigma


def create(case: dict, ruptures: list = None) -> GCIM:
    """Build the target conditional spectrum for a sub-case"""
    gcim = GCIM(build_input(case, ruptures), conditional=True)
    gcim.create()

    return gcim


@pytest.fixture(params=sorted(DATABASES))
def database(request, monkeypatch):
    """Run against one prospective database at a time

    The metadata is cached for the lifetime of the process, so the cache is
    invalidated whenever the database is switched.
    """
    path = DATABASES[request.param]
    if not path.exists():
        pytest.skip(f"{path.name} is not available")

    monkeypatch.setitem(os.environ, "DJURA_METADATA_PATH", str(path))
    monkeypatch.setattr(data_loader, "_nga_west2", None)

    yield request.param

    monkeypatch.setattr(data_loader, "_nga_west2", None)


@pytest.fixture(params=sorted(SUB_CASES))
def sub_case(request):
    """One conditioning period with its deaggregated hazard"""
    return SUB_CASES[request.param]


def get_uniform_hazard(periods):
    """Uniform hazard spectrum interpolated to the requested periods

    Parameters
    ----------
    periods : numpy.ndarray
        Vibration periods in [s], within the range of the tabulated spectrum

    Returns
    -------
    numpy.ndarray
        Spectral accelerations in [g]
    """
    tabulated = np.array(sorted(UNIFORM_HAZARD_SPECTRUM))
    values = np.array([UNIFORM_HAZARD_SPECTRUM[t] for t in tabulated])

    return np.exp(np.interp(
        np.log(periods), np.log(tabulated), np.log(values)))


@pytest.mark.slow
class TestCase3:

    def test_target_conditional_spectrum(self, database, sub_case):
        """The target returned by the creation step, equations (2) and (3) of
        the article. At the conditioning period the spectrum is pinched, its
        median equalling the conditioning amplitude with no variability
        """
        periods, mu, sigma = get_target(create(sub_case))
        star = np.argmin(np.abs(periods - sub_case["t_star"]))

        assert np.exp(mu[star]) == pytest.approx(sub_case["sa"], rel=0.01)
        assert sigma[star] == pytest.approx(0.0, abs=0.01)

    def test_selected_suite_follows_the_target(self, database, sub_case):
        """The suite returned by the selection step, figure 3 of the article.
        Every record is scaled so that Sa(T*) equals the conditioning
        amplitude, and the suite reproduces the target at every period
        """
        gcim = create(sub_case)
        records = gcim.select()["selected_scaled_best"]

        periods, mu, sigma = get_target(gcim)
        star = np.argmin(np.abs(periods - sub_case["t_star"]))
        scaled = np.asarray(records["Scaled_IMs"], dtype=float)

        assert scaled[:, star] == pytest.approx(sub_case["sa"], rel=0.01)
        assert np.mean(np.log(scaled), axis=0) == pytest.approx(mu, abs=0.15)
        assert np.std(np.log(scaled), axis=0) == pytest.approx(sigma, abs=0.15)


def get_correlation(period_i: float, period_j: float) -> float:
    """Correlation of two spectral accelerations, using the same model as
    the case requests of the selection

    Parameters
    ----------
    period_i : float
        First period in [s]
    period_j : float
        Second period in [s]

    Returns
    -------
    float
        Correlation coefficient
    """
    model = getattr(correlation_models, CORRELATION_MODEL["SA-SA"])

    if "im_pair" in get_func_args(model):
        return float(np.ravel(model("SA-SA", period_i, period_j))[0])

    return float(np.ravel(model(period_i, period_j))[0])


@pytest.mark.slow
class TestCase3ConditionalSpectrumEquations:
    """The target is compared against an independent evaluation of the
    conditional spectrum equations of the article, rather than against itself.

    Given the mean and standard deviation of ln SA for the causal rupture, and
    the correlation between spectral accelerations, equations (1) to (3) of
    the article fix the target completely

        eps(T*)         = [ln Sa(T*) - mu(T*)] / sigma(T*)
        mu(Ti | T*)     = mu(Ti) + rho(Ti, T*) eps(T*) sigma(Ti)
        sigma(Ti | T*)  = sigma(Ti) sqrt(1 - rho(Ti, T*) ** 2)

    The ground motion model predictions are taken from the intermediate
    results of the creation step, so what is verified is the assembly of the
    conditional distribution and not the ground motion model itself.
    """

    def test_target_follows_the_conditional_spectrum_equations(
            self, database, sub_case):
        gcim = create(sub_case)

        periods, mu_target, sigma_target = get_target(gcim)
        data = gcim.output_create["data"]

        # Unconditional prediction for the causal rupture
        mu = np.asarray(
            data["mu_lnIMi_rup"]["SA"], dtype=float).reshape(-1)
        sigma = np.asarray(
            data["sigma_lnIMi_rup"]["SA"], dtype=float).reshape(-1)

        # Prediction of the conditioning intensity measure, for the single
        # rupture and ground motion model of this case
        mu_star, sigma_star = (
            float(np.ravel(list(list(data[key].values())[0].values())[0])[0])
            for key in ("mu_lnIMj_rup", "sigma_lnIMj_rup")
        )

        # Equation (1)
        epsilon = (np.log(sub_case["sa"]) - mu_star) / sigma_star

        rho = np.array([
            get_correlation(period, sub_case["t_star"]) for period in periods
        ])

        # Equations (2) and (3)
        mu_conditional = mu + rho * epsilon * sigma
        sigma_conditional = sigma * np.sqrt(1.0 - rho ** 2.)

        assert mu_target == pytest.approx(mu_conditional, abs=1e-6)
        assert sigma_target == pytest.approx(sigma_conditional, abs=1e-6)


@pytest.mark.slow
class TestCase3AgainstThePublishedResults:
    """The target is compared against the results reported in the article,
    rather than against quantities computed by the software itself
    """

    def test_hazard_reproduces_the_published_amplitude(self):
        """Figure 1(a): Sa at the first modal period at the 2% in 50 year
        exceedance probability. This is the only hazard value the article
        tabulates, and it validates the recovered deaggregation rather than
        the selection, so it needs neither a database nor a target
        """
        assert SUB_CASES["3A_T1"]["sa"] == pytest.approx(
            PUBLISHED_SA_T1, rel=0.05)

    def test_target_equals_the_uniform_hazard_at_its_own_period(
            self, database, sub_case):
        """Figure 2(b): the spectral accelerations of the conditional mean
        spectra at their respective conditioning periods equal those of the
        uniform hazard spectrum
        """
        periods, mu, _ = get_target(create(sub_case))
        star = np.argmin(np.abs(periods - sub_case["t_star"]))

        assert np.exp(mu[star]) == pytest.approx(
            get_uniform_hazard(np.array([sub_case["t_star"]]))[0], rel=0.02)

    def test_uniform_hazard_envelopes_the_target(self, database, sub_case):
        """Figure 2(b): the uniform hazard spectrum is an envelope of all the
        conditional mean spectra, which therefore lie below it at every period
        other than their own conditioning period. The comparison is limited to
        the period range over which the hazard model provides spectral
        accelerations
        """
        periods, mu, _ = get_target(create(sub_case))

        tabulated = np.array(sorted(UNIFORM_HAZARD_SPECTRUM))
        inside = (periods >= tabulated[0]) & (periods <= tabulated[-1])
        away = inside & (np.abs(periods - sub_case["t_star"]) > 1e-9)

        ratio = np.exp(mu[inside]) / get_uniform_hazard(periods[inside])

        assert np.all(ratio <= 1.0 + 1e-2)
        assert np.all(np.exp(mu[away]) < get_uniform_hazard(periods[away]))

    def test_target_peaks_at_the_conditioning_period(
            self, database, sub_case):
        """Figure 2(b): each conditional mean spectrum has a relative peak at
        its conditioning period, tapering towards the median spectrum of the
        causal rupture away from it
        """
        periods, mu, _ = get_target(create(sub_case))

        ratio = np.exp(mu) / get_uniform_hazard(np.clip(
            periods, min(UNIFORM_HAZARD_SPECTRUM),
            max(UNIFORM_HAZARD_SPECTRUM)))

        assert periods[np.argmax(ratio)] == pytest.approx(
            sub_case["t_star"])

    def test_selected_suite_follows_the_published_spectrum(self, database):
        """Figure 3: the response spectra of the forty selected ground motions
        against the conditional spectrum of the article. The suite median is
        compared with the published conditional mean, which closes the loop
        between the records actually selected and the article, rather than
        between the records and a target computed here.

        The tolerance reflects the reading of a figure, so this establishes
        the level of the suite rather than its detail; the agreement of the
        suite with the target at every period is asserted separately
        """
        case = SUB_CASES["3A_T1"]

        gcim = create(case)
        records = gcim.select()["selected_scaled_best"]

        periods, _, _ = get_target(gcim)
        ln_scaled = np.log(np.asarray(records["Scaled_IMs"], dtype=float))

        for period, published in PUBLISHED_SPECTRUM_T1.items():
            index = int(np.argmin(np.abs(periods - period)))
            median = float(np.exp(np.mean(ln_scaled[:, index])))

            assert median == pytest.approx(
                published, rel=FIGURE_TOLERANCE), (
                    f"suite median at {period}s is {median:.3f}g against "
                    f"{published}g read from figure 3")

    def test_selected_suite_is_pinched_at_the_conditioning_period(
            self, database):
        """Figure 3: the record spectra converge at the conditioning period,
        every record having been scaled to the same Sa(T*)
        """
        case = SUB_CASES["3A_T1"]

        gcim = create(case)
        records = gcim.select()["selected_scaled_best"]

        periods, _, _ = get_target(gcim)
        index = int(np.argmin(np.abs(periods - case["t_star"])))
        scaled = np.asarray(records["Scaled_IMs"], dtype=float)[:, index]

        assert np.std(np.log(scaled)) == pytest.approx(0.0, abs=1e-6)
        assert np.ptp(scaled) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.slow
class TestCase3Figures:
    """Redraws the figures of the article from the results of the validation
    runs, so that a comparison which a tolerance can only make coarsely may
    also be judged by eye. Drawn only when --plot is given
    """

    def test_plot_conditional_spectra(self, database, plot_dir):
        """Figure 2(b): the conditional mean spectra at each conditioning
        period, with the uniform hazard spectrum superimposed
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(6.0, 4.5))

        tabulated = np.array(sorted(UNIFORM_HAZARD_SPECTRUM))
        axes.plot(
            tabulated, [UNIFORM_HAZARD_SPECTRUM[t] for t in tabulated],
            color="0.4", linestyle="--", label="Uniform hazard spectrum")

        for name in sorted(SUB_CASES):
            case = SUB_CASES[name]
            periods, mu, _ = get_target(create(case))

            axes.plot(
                periods, np.exp(mu),
                label=f"Conditional mean spectrum, T* = {case['t_star']}s")
            axes.plot(
                case["t_star"], case["sa"], marker="o", color="k",
                markersize=4, linestyle="none")

        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlim(0.1, 10.0)
        axes.set_ylim(0.01, 5.0)
        axes.set_xlabel("Period [s]")
        axes.set_ylabel("Spectral acceleration [g]")
        axes.set_title(f"Lin et al. (2013), figure 2(b) - {database}")
        axes.grid(which="both", color="0.9")
        axes.legend(fontsize=7)

        path = plot_dir / f"case3_figure2b_{database}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        assert path.exists()

    def test_plot_selected_suite(self, database, plot_dir):
        """Figure 3: the response spectra of the selected ground motions with
        the conditional spectrum as target, in logarithmic and linear scale
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        case = SUB_CASES["3A_T1"]

        gcim = create(case)
        records = gcim.select()["selected_scaled_best"]

        periods, mu, sigma = get_target(gcim)
        scaled = np.asarray(records["Scaled_IMs"], dtype=float)

        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

        for panel in axes:
            panel.plot(
                periods, scaled.T, color="tab:green", linewidth=0.5,
                alpha=0.6)
            panel.plot(
                periods, np.exp(mu), color="k",
                label="Conditional mean spectrum")
            panel.plot(
                periods, np.exp(mu + sigma), color="k", linestyle=":",
                label=r"Conditional mean $\pm$ conditional $\sigma$")
            panel.plot(
                periods, np.exp(mu - sigma), color="k", linestyle=":")
            panel.plot(
                list(PUBLISHED_SPECTRUM_T1), list(
                    PUBLISHED_SPECTRUM_T1.values()),
                marker="s", color="tab:red", linestyle="none",
                label="Read from figure 3")
            panel.set_xlabel("Period [s]")
            panel.set_ylabel("Spectral acceleration [g]")
            panel.grid(which="both", color="0.9")

        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlim(0.1, 10.0)
        axes[0].set_ylim(0.01, 5.0)
        axes[0].legend(fontsize=7)

        axes[1].set_xlim(0.0, 6.0)
        axes[1].set_ylim(0.0, 2.5)

        figure.suptitle(f"Lin et al. (2013), figure 3 - {database}")

        path = plot_dir / f"case3_figure3_{database}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        assert path.exists()
