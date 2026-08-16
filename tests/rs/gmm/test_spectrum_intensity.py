"""
Verification of the indirect spectrum intensity ground motion models
against the examples published in the articles they implement.

All of the published scenarios use the Boore and Atkinson (2008) GMPE for
the spectral accelerations and the Baker and Jayaram (2008) correlation
model, at a site with a 30 m weighted-average shear-wave velocity of
300 m/s subjected to a strike-slip rupture.

The target values are read off the figures of the articles, hence the
medians are only compared to within a few percent. The dispersions are
insensitive to the unstated details of the rupture scenario (Rjb against
Rrup, style of faulting flag) and are therefore compared more tightly.
"""

import pytest

from numpy import exp, ravel

from djura.record_selection.gsim.oq import OQ
from djura.record_selection.gsim.imt import ASI, SI, DSI
from djura.record_selection.gsim.models import (
    BooreAtkinson2008,
    Bradley2010ASI,
    BradleyEtAl2009SI,
    Bradley2011DSI,
)


#: Width of the period range over which DSI is integrated, in [s]
DSI_RANGE_WIDTH = 3.0


def get_context(mag: float, rjb: float, vs30: float = 300.0):
    """Strike-slip rupture scenario at a site of a given shear-wave
    velocity
    """
    return OQ()._set_contexts(
        {"mag": mag, "rjb": rjb, "rake": 0.0, "dip": 90.0, "vs30": vs30})


def get_prediction(gmpe, ctx, imt):
    """Median and total lognormal standard deviation of the intensity
    measure for a single site
    """
    mean, stddevs = gmpe.get_mean_and_stddevs(ctx, imt())

    return exp(mean)[0], ravel(stddevs[0])[0]


class TestBradley2010ASI:
    """
    Bradley, B. A. (2010). Site-Specific and Spatially Distributed
    Ground-Motion Prediction of Acceleration Spectrum Intensity.
    Bulletin of the Seismological Society of America, 100(2), 792-801.
    """

    @pytest.mark.parametrize(
        "mag, rjb, median, sigma", [
            # Converged values of fig. 2a and 2b
            (6.5, 30.0, 0.111, 0.549),
            (6.0, 20.0, 0.094, 0.5475),
        ]
    )
    @pytest.mark.parametrize("spacing", ["linear", "log"])
    def test_figure_2(self, mag, rjb, median, sigma, spacing):
        """Nine integration points are appropriate for a wide range of
        magnitude and distance scenarios, with both spacings converging
        to the same prediction
        """
        gmpe = Bradley2010ASI(
            BooreAtkinson2008(), "baker_jayaram", n_per=9, spacing=spacing)

        asi, sig = get_prediction(gmpe, get_context(mag, rjb), ASI)

        assert asi == pytest.approx(median, rel=0.05)
        assert sig == pytest.approx(sigma, rel=0.02)

    @pytest.mark.parametrize("spacing", ["linear", "log"])
    def test_convergence(self, spacing):
        """Convergence is achieved as the number of integration points
        increases, fig. 2
        """
        ctx = get_context(6.5, 30.0)

        coarse = get_prediction(Bradley2010ASI(
            BooreAtkinson2008(), n_per=9, spacing=spacing), ctx, ASI)
        fine = get_prediction(Bradley2010ASI(
            BooreAtkinson2008(), n_per=81, spacing=spacing), ctx, ASI)

        assert coarse[0] == pytest.approx(fine[0], rel=0.01)
        assert coarse[1] == pytest.approx(fine[1], rel=0.01)

    def test_period_range(self):
        """ASI is defined over the 0.1-0.5s period range, eq. (1)"""
        gmpe = Bradley2010ASI(BooreAtkinson2008(), n_per=9)

        assert gmpe.periods[0] == pytest.approx(0.1)
        assert gmpe.periods[-1] == pytest.approx(0.5)
        assert len(gmpe.periods) == 9
        assert sum(gmpe.weights) == pytest.approx(0.4)


class TestBradleyEtAl2009SI:
    """
    Bradley, B. A., Cubrinovski, M., MacRae, G. A., & Dhakal, R. P. (2009).
    Ground-Motion Prediction Equation for SI Based on Spectral Acceleration
    Equations. Bulletin of the Seismological Society of America, 99(1),
    277-285.
    """

    @pytest.mark.parametrize(
        "mag, rjb, median, sigma", [
            # Converged values of fig. 4a and 4b
            (6.5, 30.0, 42.8, 0.578),
            (6.0, 20.0, 30.7, 0.5715),
        ]
    )
    def test_figure_4(self, mag, rjb, median, sigma):
        """A discretisation step below 0.2s is appropriate for a wide range
        of magnitude and distance scenarios
        """
        gmpe = BradleyEtAl2009SI(
            BooreAtkinson2008(), "baker_jayaram", delta_t=0.1)

        si, sig = get_prediction(gmpe, get_context(mag, rjb), SI)

        assert si == pytest.approx(median, rel=0.05)
        assert sig == pytest.approx(sigma, rel=0.02)

    def test_coarse_discretisation(self):
        """A step-size of 1.2s underpredicts the median and overpredicts
        the dispersion, fig. 4
        """
        ctx = get_context(6.5, 30.0)

        coarse = get_prediction(BradleyEtAl2009SI(
            BooreAtkinson2008(), delta_t=1.2), ctx, SI)
        fine = get_prediction(BradleyEtAl2009SI(
            BooreAtkinson2008(), delta_t=0.1), ctx, SI)

        assert coarse[0] < fine[0]
        assert coarse[1] > fine[1]
        assert coarse[0] == pytest.approx(33.5, rel=0.05)
        assert coarse[1] == pytest.approx(0.611, rel=0.02)

    def test_convergence(self):
        """Convergence is achieved as the discretisation step is reduced,
        fig. 4
        """
        ctx = get_context(6.5, 30.0)

        coarse = get_prediction(BradleyEtAl2009SI(
            BooreAtkinson2008(), delta_t=0.1), ctx, SI)
        fine = get_prediction(BradleyEtAl2009SI(
            BooreAtkinson2008(), delta_t=0.01), ctx, SI)

        assert coarse[0] == pytest.approx(fine[0], rel=0.01)
        assert coarse[1] == pytest.approx(fine[1], rel=0.01)

    def test_period_range(self):
        """SI is defined over the 0.1-2.5s period range, eq. (1), giving 25
        terms in the summation for a discretisation step of 0.1s, p. 280
        """
        gmpe = BradleyEtAl2009SI(BooreAtkinson2008(), delta_t=0.1)

        assert gmpe.periods[0] == pytest.approx(0.1)
        assert gmpe.periods[-1] == pytest.approx(2.5)
        assert len(gmpe.periods) == 25
        assert sum(gmpe.weights) == pytest.approx(2.4)


class TestBradley2011DSI:
    """
    Bradley, B. A. (2011). Empirical equations for the prediction of
    displacement spectrum intensity and its correlation with other
    intensity measures. Soil Dynamics and Earthquake Engineering, 31(8),
    1182-1191.
    """

    @pytest.mark.parametrize(
        "mag, rjb, median, sigma", [
            # Converged values of fig. 3a and 3b
            (7.0, 30.0, 13.0, 0.672),
            (6.5, 20.0, 9.4, 0.670),
        ]
    )
    @pytest.mark.parametrize("spacing", ["linear", "log"])
    def test_figure_3(self, mag, rjb, median, sigma, spacing):
        """Nine integration points are appropriate for a wide range of
        magnitude and distance scenarios, with the linear and logarithmic
        spacings of periods giving essentially identical predictions.

        The medians of fig. 3a are the spectral displacements averaged over
        the 2.0-5.0s period range rather than integrated over it, and are
        therefore compared against the integral of eq. (1) divided by the
        width of the period range
        """
        gmpe = Bradley2011DSI(
            BooreAtkinson2008(), "baker_jayaram", n_per=9, spacing=spacing)

        dsi, sig = get_prediction(gmpe, get_context(mag, rjb), DSI)

        assert dsi / DSI_RANGE_WIDTH == pytest.approx(median, rel=0.05)
        assert sig == pytest.approx(sigma, rel=0.02)

    @pytest.mark.parametrize("spacing", ["linear", "log"])
    def test_convergence(self, spacing):
        """Convergence is achieved as the number of integration points
        increases, fig. 3
        """
        ctx = get_context(7.0, 30.0)

        coarse = get_prediction(Bradley2011DSI(
            BooreAtkinson2008(), n_per=9, spacing=spacing), ctx, DSI)
        fine = get_prediction(Bradley2011DSI(
            BooreAtkinson2008(), n_per=81, spacing=spacing), ctx, DSI)

        assert coarse[0] == pytest.approx(fine[0], rel=0.01)
        assert coarse[1] == pytest.approx(fine[1], rel=0.01)

    def test_period_range(self):
        """DSI is defined over the 2.0-5.0s period range, eq. (1)"""
        gmpe = Bradley2011DSI(BooreAtkinson2008(), n_per=9)

        assert gmpe.periods[0] == pytest.approx(2.0)
        assert gmpe.periods[-1] == pytest.approx(5.0)
        assert len(gmpe.periods) == 9
        assert sum(gmpe.weights) == pytest.approx(DSI_RANGE_WIDTH)


class TestSpectrumIntensityCommon:
    """
    Behaviour shared by the three spectrum intensity models
    """

    @pytest.mark.parametrize(
        "gmpe_class, imt", [
            (Bradley2010ASI, ASI),
            (BradleyEtAl2009SI, SI),
            (Bradley2011DSI, DSI),
        ]
    )
    def test_standard_deviation_components(self, gmpe_class, imt):
        """The inter- and intra-event standard deviations are propagated
        whenever the underlying model provides them, and combine into the
        total standard deviation
        """
        gmpe = gmpe_class(BooreAtkinson2008())

        _, stddevs = gmpe.get_mean_and_stddevs(get_context(6.5, 30.0), imt())

        assert len(stddevs) == 3

        sig, tau, phi = (ravel(stddev)[0] for stddev in stddevs)

        assert sig ** 2. == pytest.approx(tau ** 2. + phi ** 2.)
        assert phi < sig

    @pytest.mark.parametrize(
        "gmpe_class, imt", [
            (Bradley2010ASI, ASI),
            (BradleyEtAl2009SI, SI),
            (Bradley2011DSI, DSI),
        ]
    )
    def test_inherited_parameters(self, gmpe_class, imt):
        """The parameters required by the underlying spectral acceleration
        model are those required by the spectrum intensity model
        """
        gmpe = gmpe_class(BooreAtkinson2008())

        assert gmpe.REQUIRES_RUPTURE_PARAMETERS == \
            BooreAtkinson2008.REQUIRES_RUPTURE_PARAMETERS
        assert gmpe.REQUIRES_SITES_PARAMETERS == \
            BooreAtkinson2008.REQUIRES_SITES_PARAMETERS
        assert gmpe.REQUIRES_DISTANCES == BooreAtkinson2008.REQUIRES_DISTANCES
        assert gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES == {imt}

    @pytest.mark.parametrize(
        "gmpe_class", [Bradley2010ASI, BradleyEtAl2009SI, Bradley2011DSI])
    def test_invalid_correlation_function(self, gmpe_class):
        with pytest.raises(ValueError) as exc:
            gmpe_class(BooreAtkinson2008(), "not_a_correlation_function")

        assert str(exc.value) == "Not a valid correlation function"

    @pytest.mark.parametrize(
        "gmpe_class", [Bradley2010ASI, Bradley2011DSI])
    @pytest.mark.parametrize("n_per, spacing", [(2, "log"), (9, "quadratic")])
    def test_invalid_discretisation_by_count(
            self, gmpe_class, n_per, spacing):
        with pytest.raises(ValueError):
            gmpe_class(BooreAtkinson2008(), n_per=n_per, spacing=spacing)

    @pytest.mark.parametrize("delta_t", [0.0, -0.1, 2.0])
    def test_invalid_discretisation_by_step(self, delta_t):
        with pytest.raises(ValueError):
            BradleyEtAl2009SI(BooreAtkinson2008(), delta_t=delta_t)
