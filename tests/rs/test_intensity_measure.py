import pytest
from pathlib import Path
import numpy as np

from djura.record_selection import intensity_measure as im_module
from djura.record_selection.intensity_measure import IntensityMeasure

path = Path(__file__).resolve().parent


class TestIntensityMeasure:
    g = 9.81

    @pytest.fixture(scope="module")
    def model(self):
        return IntensityMeasure()

    @pytest.fixture(scope="module")
    def record(self):
        record_filename = path / "assets/records/RSN288_ITALY_A-BRZ000.AT2"
        ug = np.loadtxt(record_filename).transpose()

        return ug

    @pytest.mark.parametrize(
        "sa, period, sd", [
            (0.3, 0.5, 0.019),
            (0.5, 1.0, 0.124),
            (1.0, 4.0, 3.976),
            (0.0025, 1.0, 0.001),
        ]
    )
    def test_sa_sd(self, model: IntensityMeasure, sa, period, sd):
        tol_sa = 0.1
        tol_sd = 0.001

        sa_computed = model.sd_to_sa(sd, period)
        sd_computed = model.sa_to_sd(sa, period)

        assert sa == pytest.approx(sa_computed, abs=tol_sa)
        assert sd == pytest.approx(sd_computed, abs=tol_sd)

    @pytest.mark.parametrize(
        "dt, tn, alpha, beta, expected", [
            (0.005, 1.0, 0.7, 0.85, 0.43),
            (0.005, 0.5, 0.7, 0.85, 0.48),
            (0.005, 4.0, 0.7, 0.85, 0.26),
            (0.005, 1.0, 0.7, 1.00, 0.45),
            (0.005, 0.5, 0.7, 1.00, 0.51),
            (0.005, 4.0, 0.7, 1.00, 0.30),
            (0.005, 1.0, 0.3, 2.00, 0.48),
        ]
    )
    def test_get_fiv3(self, model: IntensityMeasure, record, dt, tn, alpha,
                      beta, expected):

        out = model.get_fiv3(record, dt, tn, alpha, beta)
        fiv3 = out[0]

        assert fiv3 == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, period, damping, expected", [
            (0.005, 1.0, 0.02, 0.29),
            (0.010, 0.5, 0.05, 0.99),
            (0.005, 4.0, 0.10, 0.03),
            (0.005, 1.0, 0.05, 0.19),
            (0.005, 0.0, 0.05, 0.22),
            (0.005, 0.0, 0.02, 0.22),
            (0.010, 0.0, 0.02, 0.22),
            (0.005, 1, 0.02, 0.29),
            (0.0, 0.0, 0.02, 0.22),
            (0.0, 0.5, 0.02, 0.22),
        ]
    )
    def test_get_sat(self, model: IntensityMeasure, record, dt, period,
                     damping, expected):

        if dt == 0 and isinstance(period, float) and period != 0.0:
            with pytest.raises(ValueError) as exc:
                sa = model.get_sat(period, record, dt, damping)
                assert str(exc.value) == "Time step must not be zero!"
        else:
            sa = model.get_sat(period, record, dt, damping)
            assert sa == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, period, damping, expected", [
            (0.005, 1.0, 0.05, 0.047),
            (0.005, 0.5, 0.05, 0.025),
            (0.002, 0.5, 0.02, 0.012),
        ]
    )
    def test_get_sdt(self, model: IntensityMeasure, record, dt, period,
                     damping, expected):
        sd = model.get_sdt(record, dt, period, damping)

        assert sd == pytest.approx(expected, abs=0.001)

    @pytest.mark.parametrize(
        "dt, period, damping, expected", [
            (0.005, 1.0, 0.05, 0.30),
            (0.005, 0.5, 0.05, 0.32),
            (0.002, 0.5, 0.02, 0.16),
        ]
    )
    def test_get_svt(self, model: IntensityMeasure, record, dt, period,
                     damping, expected):
        sv = model.get_svt(record, dt, period, damping)

        assert sv == pytest.approx(expected, abs=0.01)

    def test_get_pga(self, model: IntensityMeasure, record):
        pga = model.get_pga(record)

        assert pga == pytest.approx(0.22, abs=0.01)

    @pytest.mark.parametrize(
        "dt, period, damping, bounds, expected", [
            (0.005, 1.0, 0.02, [0.2, 1.5], 0.39),
            (0.005, 0.5, 0.02, [0.2, 3.0], 0.36),
            (0.005, 0.5, 0.05, [0.2, 1.5], 0.50),
            (0.005, 1.0, 0.05, [0.2, 3.0], 0.16),
            (0.005, 0.0, 0.05, [0.2, 3.0], 0.16),
        ]
    )
    def test_get_sa_avg(self, model: IntensityMeasure, record, dt, period,
                        damping, bounds, expected):
        if period == 0:
            with pytest.raises(ValueError) as exc:
                sa_avg = model.get_sa_avg(
                    record, dt, period, damping, bounds, size=10)
                assert str(exc.value) == \
                    "Conditioning period must not be zero!"
            return

        sa_avg = model.get_sa_avg(record, dt, period, damping, bounds, size=10)

        assert sa_avg == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "osc_type, expected", [
            ("sd", 0.0048),
            ("sv", 0.0354),
            ("sa", 0.3304),
            ("psa", 0.1907),
            ("psv", 0.0304)
        ]
    )
    def test_get_sat2(self, model: IntensityMeasure, record, osc_type,
                      expected):
        sat2 = model.sat2(record, 0.005, 1.0, 0.05, osc_type)

        assert sat2 == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, expected", [
            (0.005, 0.23),
            (0.002, 0.09),
            (0.010, 0.45),
        ]
    )
    def test_get_pgv(self, model: IntensityMeasure, record, dt, expected):
        pgv = model.get_pgv(record, dt)

        assert pgv == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, expected", [
            (0.005, 0.083),
            (0.002, 0.013),
            (0.010, 0.334),
        ]
    )
    def test_get_pgd(self, model: IntensityMeasure, record, dt, expected):
        pgd = model.get_pgd(record, dt)

        assert pgd == pytest.approx(expected, abs=0.001)

    @pytest.mark.parametrize(
        "dt, expected", [
            (0.005, 0.86),
            (0.002, 0.34),
            (0.010, 1.72),
        ]
    )
    def test_get_arias_intensity(self, model: IntensityMeasure, record, dt,
                                 expected):
        ia = model.get_arias_intensity(record, dt)

        assert ia == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, damping, delta_period, expected", [
            (0.005, 0.05, 0.01, 0.265),
            (0.002, 0.05, 0.01, 0.148),
            (0.010, 0.05, 0.01, 0.232),
            (0.005, 0.02, 0.01, 0.357),
            (0.005, 0.10, 0.01, 0.197),
            (0.005, 0.05, 0.005, 0.265),
            (0.005, 0.05, 0.050, 0.270),
        ]
    )
    def test_get_asi(self, model: IntensityMeasure, record, dt, damping,
                     delta_period, expected):
        asi = model.get_asi(record, dt, damping, delta_period)

        assert asi == pytest.approx(expected, abs=0.001)

    @pytest.mark.parametrize(
        "dt, damping, delta_period, expected", [
            (0.005, 0.05, 0.01, 80.86),
            (0.002, 0.05, 0.01, 29.23),
            (0.010, 0.05, 0.01, 152.70),
            (0.005, 0.02, 0.01, 107.46),
            (0.005, 0.10, 0.01, 60.03),
            (0.005, 0.05, 0.005, 80.87),
            (0.005, 0.05, 0.050, 81.15),
        ]
    )
    def test_get_si(self, model: IntensityMeasure, record, dt, damping,
                    delta_period, expected):
        si = model.get_si(record, dt, damping, delta_period)

        assert si == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, damping, delta_period, expected", [
            (0.005, 0.05, 0.01, 46.06),
            (0.002, 0.05, 0.01, 8.26),
            (0.010, 0.05, 0.01, 114.44),
            (0.005, 0.02, 0.01, 58.68),
            (0.005, 0.10, 0.01, 36.03),
            (0.005, 0.05, 0.005, 46.06),
            (0.005, 0.05, 0.050, 46.06),
        ]
    )
    def test_get_dsi(self, model: IntensityMeasure, record, dt, damping,
                     delta_period, expected):
        dsi = model.get_dsi(record, dt, damping, delta_period)

        assert dsi == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "intensity_measure, delta_period, upper_bound", [
            ("get_asi", 0.0, 0.2),
            ("get_asi", 0.3, 0.2),
            ("get_si", -0.01, 1.2),
            ("get_si", 1.5, 1.2),
            ("get_dsi", 0.0, 1.5),
            ("get_dsi", 2.0, 1.5),
        ]
    )
    def test_spectrum_intensity_period_step(
        self, model: IntensityMeasure, record, intensity_measure,
        delta_period, upper_bound
    ):
        with pytest.raises(ValueError) as exc:
            getattr(model, intensity_measure)(
                record, 0.005, 0.05, delta_period)

        assert str(exc.value) == \
            f"Period step-size must be within (0, {upper_bound}], " \
            f"{delta_period} was given"

    @pytest.mark.parametrize(
        "dt, expected", [
            (0.005, 1.03),
            (0.002, 0.41),
            (0.010, 2.05),
        ]
    )
    def test_get_cav(self, model: IntensityMeasure, record, dt, expected):
        cav = model.get_cav(record, dt)

        assert cav == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "dt, start, end, expected", [
            (0.005, 0.05, 0.95, 17.78),
            (0.005, 0.05, 0.75, 6.87),
            (0.002, 0.05, 0.75, 2.75),
        ]
    )
    def test_get_significant_duration(self, model: IntensityMeasure, record,
                                      dt, start, end, expected):
        d = model.get_significant_duration(record, dt, start, end)

        assert d[0] == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "filename1, filename2, dt, period, damping, percentiles, num_theta, "
        "expected", [
            ("RSN288_ITALY_A-BRZ000.AT2", "RSN156_ITALY_F-CSC-NS.AT2",
             0.005, 1.0, 0.05, None, None, [0.19, 0.25, 0.35]),
            ("RSN156_ITALY_F-CSC-NS.AT2", "RSN288_ITALY_A-BRZ000.AT2",
             0.005, 0.5, 0.02, None, 180, [0.43, 0.49, 0.59]),
            ("RSN288_ITALY_A-BRZ000.AT2", "RSN156_ITALY_F-CSC-NS.AT2",
             0.005, 1.0, 0.05, [50], None, [0.25]),
            ("RSN288_ITALY_A-BRZ000.AT2", "RSN156_ITALY_F-CSC-NS.AT2",
             0.005, 1.0, 0.05, [16, 50], 100, [0.19, 0.25]),
            ("RSN288_ITALY_A-BRZ000.AT2", "RSN156_ITALY_F-CSC-NS.AT2",
             0.005, 1.0, 0.05, 16, 100, [0.19]),
        ]
    )
    def test_get_sa_rot_d_xx(self, model: IntensityMeasure, filename1,
                             filename2, dt, period, damping, percentiles,
                             num_theta, expected):
        record_filename = path / f"assets/records/{filename1}"
        acc1 = np.loadtxt(record_filename).transpose()
        record_filename = path / f"assets/records/{filename2}"
        acc2 = np.loadtxt(record_filename).transpose()

        rot_d_xx = model.get_sa_rot_d_xx(
            acc1, acc2, dt, period, damping, percentiles, num_theta)

        assert list(np.round(rot_d_xx, 2)) == expected

    def test_get_ei(self, model: IntensityMeasure, record):
        ei = model.get_ei(record, 0.005, 1.0, 0.05)

        assert ei == pytest.approx(0.1932, abs=0.01)


class TestSaMethod:
    """The non-default oscillator solver selected by ``SA_METHOD``.

    The Nigam-Jennings default is exercised by the tests above."""

    @pytest.fixture
    def model(self):
        return IntensityMeasure()

    @pytest.fixture
    def record(self):
        return np.loadtxt(
            path / "assets/records/RSN288_ITALY_A-BRZ000.AT2").transpose()

    @pytest.fixture
    def fft(self, monkeypatch):
        monkeypatch.setattr(im_module, "SA_METHOD", "fft")

    def test_solvers_agree_on_a_resolved_period(self, model, record):
        """Both solvers integrate the same oscillator, so they must agree
        wherever the time step resolves the response"""
        solved = {}
        for method in im_module.SA_METHODS:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(im_module, "SA_METHOD", method)
                solved[method] = model.get_sat(1.0, record, 0.005, 0.05)

        assert solved["nigam_jennings"] == pytest.approx(
            solved["fft"], rel=0.01)

    @pytest.mark.usefixtures("fft")
    def test_zero_period_returns_pga(self, model, record):
        assert model.get_sat(0.0, record, 0.005, 0.05) == \
            pytest.approx(np.max(np.abs(record)))

    @pytest.mark.usefixtures("fft")
    def test_vector_and_scalar_periods_match(self, model, record):
        periods = np.array([0.02, 0.2, 2.0])
        vector = model.get_sat(periods, record, 0.005, 0.05)
        scalar = [model.get_sat(t, record, 0.005, 0.05) for t in periods]

        assert isinstance(vector, np.ndarray)
        assert vector == pytest.approx(scalar)

    @pytest.mark.usefixtures("fft")
    def test_derived_measures_follow_the_solver(self, model, record):
        """Sd and Sv are scaled from Sa, so they must track the switch"""

        sa = model.get_sat(1.0, record, 0.005, 0.05)
        omega = 2 * np.pi

        assert model.get_sdt(record, 0.005, 1.0, 0.05) == \
            pytest.approx(sa * model.g / omega ** 2)
        assert model.get_svt(record, 0.005, 1.0, 0.05) == \
            pytest.approx(sa * model.g / omega)

    def test_unknown_solver_raises(self, model, record, monkeypatch):
        monkeypatch.setattr(im_module, "SA_METHOD", "duhamel")

        with pytest.raises(ValueError, match="Unknown oscillator solver"):
            model.get_sat(1.0, record, 0.005, 0.05)
