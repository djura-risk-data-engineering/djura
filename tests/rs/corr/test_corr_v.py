import pytest

import djura.record_selection.correlation_models_v as corr_v


class TestVerticalCorrelations:
    @pytest.mark.parametrize(
        "t1, t2, mag, region, total, between, within", [
            # Equal periods -> perfect correlation in all components
            (1.0, 1.0, 6.5, "CAL", 1.0, 1.0, 1.0),
            # On-grid pairs -> within/between equal the Table 4/5 cells
            (0.2, 1.0, 6.5, "CAL", 0.432, 0.384, 0.453),
            (0.05, 0.15, 6.5, "CAL", 0.775, 0.843, 0.742),
            # Off-grid (log-period interpolation)
            (0.3, 3.0, 7.0, "CAL", 0.330, 0.424, 0.328),
            # Japan uses a different sigma split -> different total only
            (0.5, 2.0, 6.5, "JPN", 0.532, 0.692, 0.502),
        ]
    )
    def test_gkas2017_v(self, t1, t2, mag, region, total, between, within):
        rho_total, rho_between, rho_within = corr_v.gkas2017_v(
            t1, t2, mag, region)

        assert rho_total == pytest.approx(total, abs=0.01)
        assert rho_between == pytest.approx(between, abs=0.01)
        assert rho_within == pytest.approx(within, abs=0.01)

    @pytest.mark.parametrize("t1, t2", [(0.005, 1.0), (1.0, 12.0)])
    def test_period_out_of_range(self, t1, t2):
        with pytest.raises(ValueError):
            corr_v.gkas2017_v(t1, t2, 6.5)

    def test_invalid_region(self):
        with pytest.raises(ValueError):
            corr_v.gkas2017_v(1.0, 2.0, 6.5, region="XXX")
