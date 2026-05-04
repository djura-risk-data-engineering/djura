import pytest
from pathlib import Path
import numpy as np

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
        "dt, expected", [
            (0.005, 10.1),
            (0.002, 4.0),
            (0.010, 20.1),
        ]
    )
    def test_get_cav(self, model: IntensityMeasure, record, dt, expected):
        cav = model.get_cav(record, dt)

        assert cav == pytest.approx(expected, abs=0.1)

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
