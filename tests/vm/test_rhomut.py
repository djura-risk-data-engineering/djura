import numpy as np
import pytest

from djura.edp_im.r_mu_t import (
    ec8, guerrini, krawinkler_nassar, miranda, newmark_hall, vidic,
)


class TestRhomut:
    @pytest.mark.parametrize("T, R", [
        (0.5, 2.0),
        (1.0, 3.0),
        (1.5, 3.0),
    ])
    def test_ec8(self, T, R):
        result = ec8.make_prediction(
            dynamic_ductility=3, period=T, period_c=1)
        assert result["median"] == R

    @pytest.mark.parametrize("case, ahyst, Thyst", [
        ("FD", 0.7, 0.055),
        ("IN", 0.2, 0.030),
        ("SD", 0.0, 0.022),
    ])
    def test_guerrini(self, case, Thyst, ahyst):
        result = guerrini.make_prediction(
            strength_ratio=4.0, period=1, case=case, period_c=0.8)

        expected = abs(4 + pow(4 - 1, 2.1)
                       / ((1.0 / Thyst + ahyst) * pow(1.0 / 0.8, 2.3)))

        assert result["median"] == expected

    @pytest.mark.parametrize("ah, r", [
        (0, 3.11),
        (2, 3.18),
        (10, 3.32),
    ])
    def test_krawinkler_nassar(self, ah, r):
        result = krawinkler_nassar.make_prediction(
            dynamic_ductility=3.0, ah=ah, period=1)
        assert result["median"] == pytest.approx(r, abs=0.01)

    @pytest.mark.parametrize("site, r", [
        ("rock", 3.35),
        ("soft-soil", 3.70),
        ("alluvium", 3.81),
    ])
    def test_miranda(self, site, r):
        result = miranda.make_prediction(
            dynamic_ductility=3.0, site=site, period=1, period_g=1)
        assert result["median"] == pytest.approx(r, abs=0.01)

    @pytest.mark.parametrize("T, R", [
        (1 / 50, 1.0),
        (0.1, "compute"),
        (0.3, 3.0),
        (0.8, 4.2),
        (1.2, 5.0),
    ])
    def test_newmark_hall(self, T, R):
        result = newmark_hall.make_prediction(
            dynamic_ductility=5.0, period=T, period_cc=0.5, period_c=1.0)

        if R == "compute":
            beta = np.log(T / (1. / 33)) / np.log(0.125 / (1. / 33))
            R = pow(2 * 5 - 1, 0.5 * beta)

        assert pytest.approx(result["median"], abs=0.01) == R

    @pytest.mark.parametrize("T, R", [
        (0.5, 2.98),
        (1.5, 5.11),
    ])
    def test_vidic(self, T, R):
        result = vidic.make_prediction(
            dynamic_ductility=5.0, period=T, period_c=1.0)

        assert R == pytest.approx(result["median"], abs=0.01)
