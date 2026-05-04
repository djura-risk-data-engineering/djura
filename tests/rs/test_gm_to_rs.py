import numpy as np
import pytest
from pathlib import Path

from djura.record_selection.gm_to_rs import ResponseSpectrumFromGM

path = Path(__file__).resolve().parent
RECORD = path / "assets/records/RSN288_ITALY_A-BRZ000.AT2"
DT = 0.005


@pytest.fixture(scope="module")
def acc():
    return np.loadtxt(RECORD)


@pytest.fixture(scope="module")
def rs():
    return ResponseSpectrumFromGM(damping=0.05)


class TestResponseSpectrumFromGM:
    def test_default_periods_length(self, rs, acc):
        periods, sa = rs.derive_response_spectrum(acc, dt=DT)
        assert len(periods) == 401
        assert len(sa) == 401

    def test_default_periods_bounds(self, rs, acc):
        periods, _ = rs.derive_response_spectrum(acc, dt=DT)
        assert periods[0] == pytest.approx(0.0)
        assert periods[-1] == pytest.approx(4.0)

    def test_returns_lists(self, rs, acc):
        periods, sa = rs.derive_response_spectrum(acc, dt=DT)
        assert isinstance(periods, list)
        assert isinstance(sa, list)

    def test_known_sa_values(self, rs, acc):
        periods, sa = rs.derive_response_spectrum(acc, dt=DT)
        # T=0.0s (PGA), T=0.5s, T=1.0s, T=2.0s
        assert sa[0] == pytest.approx(0.2196, abs=1e-3)
        assert sa[50] == pytest.approx(0.4049, abs=1e-3)
        assert sa[100] == pytest.approx(0.1907, abs=1e-3)
        assert sa[200] == pytest.approx(0.1242, abs=1e-3)

    def test_custom_periods(self, rs, acc):
        custom = [0.5, 1.0, 2.0]
        periods, sa = rs.derive_response_spectrum(acc, dt=DT, periods=custom)
        assert periods == pytest.approx(custom)
        assert sa[0] == pytest.approx(0.4049, abs=1e-3)
        assert sa[1] == pytest.approx(0.1907, abs=1e-3)
        assert sa[2] == pytest.approx(0.1242, abs=1e-3)

    def test_damping_affects_output(self, acc):
        rs5 = ResponseSpectrumFromGM(damping=0.05)
        rs2 = ResponseSpectrumFromGM(damping=0.02)
        _, sa5 = rs5.derive_response_spectrum(acc, dt=DT, periods=[0.5, 1.0])
        _, sa2 = rs2.derive_response_spectrum(acc, dt=DT, periods=[0.5, 1.0])
        assert sa2[0] == pytest.approx(0.4890, abs=1e-3)
        assert sa2[1] == pytest.approx(0.2967, abs=1e-3)
        # lower damping → higher spectral acceleration
        assert sa2[0] > sa5[0]
        assert sa2[1] > sa5[1]

    def test_output_format_stored(self):
        rs = ResponseSpectrumFromGM(damping=0.05, output_format="Dict")
        assert rs.output_format == "dict"
