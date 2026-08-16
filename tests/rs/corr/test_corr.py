import pytest

import djura.record_selection.correlation_models as corr


class TestCorrelations:
    @pytest.mark.parametrize(
        "t1, t2, out", [
            (0.5, 0.5, 1.0),
            (0.05, 0.15, 0.92),
            (0.05, 0.08, 0.96),
            (0.05, 0.8, 0.47),
        ]
    )
    def test_baker_jayaram(self, t1, t2, out):
        val = corr.baker_jayaram(t1, t2)

        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t1, t2, out", [
            (0.5, 0.5, 1.0),
            (0.05, 0.15, 0.88),
            (0.05, 0.08, 0.96),
            (0.005, 0.8, 0.51),
        ]
    )
    def test_akkar(self, t1, t2, out):

        if min(t1, t2) < 0.01 or max(t1, t2) > 4:
            with pytest.raises(ValueError) as exc:
                corr.akkar(t1, t2)

                assert str(exc.value) == "Period array contains values outside"
                " of the range supported by the Akkar et al. (2014) "
                "correlation model"

            return

        val = corr.akkar(t1, t2)

        assert val == pytest.approx(out, abs=0.01)

    def test_bradley2011_ds(self):
        assert corr.bradley2011_ds() == 0.843

    def test_bradley2011_ds_pgvs(self):
        assert corr.bradley2011_ds595_pgv() == -0.211
        assert corr.bradley2011_ds575_pgv() == -0.259

    @pytest.mark.parametrize(
        "t, out", [
            (None, -0.405),
            (11, None),
            (0.03, -0.41),
            (0.07, -0.39),
            (0.20, -0.36),
            (0.80, -0.13),
            (2.80, 0.10),
            (7.80, 0.12),
        ]
    )
    def test_bradley2011_ds595_sa(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_ds595_sa(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_ds595_sa(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t, out", [
            (None, -0.442),
            (11, None),
            (0.07, -0.40),
            (0.20, -0.39),
            (0.80, -0.18),
            (2.80, 0.04),
            (7.80, 0.09),
        ]
    )
    def test_bradley2011_ds575_sa(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_ds575_sa(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_ds575_sa(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t, out", [
            (11, None),
            (0.5, 0.74),
            (0.0, 1.0),
        ]
    )
    def test_bradley2011_pga(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_pga(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_pga(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t, out", [
            (11, None),
            # Figure 6a, the correlation peaks over the 0.1-0.5s range
            # over which ASI is defined
            (0.01, 0.93),
            (0.3, 0.96),
            (1.0, 0.59),
            (9.99, 0.30),
            # Tends to the correlation between ASI and PGA
            (0.0, 0.928),
        ]
    )
    def test_bradley2011_asi_sa(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_asi_sa(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_asi_sa(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t, out", [
            (11, None),
            # Figure 6b, the correlation peaks over the moderate period
            # range which dominates the magnitude of SI
            (0.01, 0.60),
            (0.1, 0.40),
            (1.4, 0.93),
            (9.99, 0.69),
            # Tends to the correlation between SI and PGA
            (0.0, 0.599),
        ]
    )
    def test_bradley2011_si_sa(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_si_sa(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_si_sa(t)
        assert val == pytest.approx(out, abs=0.01)

    def test_bradley2011_asi_pga(self):
        assert corr.bradley2011_asi_pga() == 0.928

    def test_bradley2011_si_pga(self):
        assert corr.bradley2011_si_pga() == 0.599

    def test_bradley2011_asi_si(self):
        assert corr.bradley2011_asi_si() == 0.641

    @pytest.mark.parametrize(
        "t, out", [
            (11, None),
            # Figure 10a, the correlation peaks over the 2.0-5.0s range
            # over which DSI is defined
            (0.01, 0.39),
            (0.15, 0.27),
            (1.0, 0.64),
            (3.4, 0.98),
            (9.99, 0.83),
            # Tends to the correlation between DSI and PGA
            (0.0, 0.395),
        ]
    )
    def test_bradley2011_dsi_sa(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2011_dsi_sa(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2011_dsi_sa(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t, out", [
            (11, None),
            (0.5, 0.77),
            (0.0, 0.73),
        ]
    )
    def test_bradley2012_pgv(self, t, out):

        if t is not None and t > 10:
            with pytest.raises(ValueError) as exc:
                corr.bradley2012_pgv(t)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.bradley2012_pgv(t)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "t1, t2, out", [
            (5, 0.2, None),
            (0.5, 5, None),
            (0.5, 0.7, 0.97),
        ]
    )
    def test_dm18(self, t1, t2, out):
        if not 0.1 <= t1 <= 3 or not 0.1 <= t2 <= 3:
            if not 0.1 <= t1 <= 3:
                t = t1
            else:
                t = t2

            with pytest.raises(ValueError) as exc:
                corr.dm18(t1, t2)

                assert str(exc.value) == f"Period ({t}) must be 0.01 <= x < 10"
            return

        val = corr.dm18(t1, t2)
        assert val == pytest.approx(out, abs=0.01)

    @pytest.mark.parametrize(
        "im_pair, t1, t2", [
            ("SA-Ds595", 0.2, None),
            ("SA-Ds575", 0.5, None),
            ("Sa_avg2-Ds595", 1.2, None),
            ("Sa_avg2-Ds575", 0.8, None),
            ("Sa_avg2-PGA", 1.2, None),
            ("Sa_avg2-PGV", 0.8, None),
            ("Sa_avg3-Ds595", 1.2, None),
            ("Sa_avg3-Ds575", 0.2, None),
            ("Sa_avg3-PGA", 1.2, None),
            ("Sa_avg3-PGV", 0.2, None),
            ("FIV3-Ds595", 1.0, None),
            ("FIV3-Ds575", 1.0, None),
            ("FIV3-PGA", 1.0, None),
            ("FIV3-PGV", 1.0, None),
            ("FIV3-PGV", 0.01, None),
            ("FIV3-FIV3", 0.2, 1.1),
            ("Sa_avg3-FIV3", 0.1, 1.3),
            ("Sa_avg2-FIV3", 0.2, 1.5),
            ("SA-Sa_avg3", 0.2, 1.2),
            ("SA-FIV3", 0.2, 1.0),
            ("SA-SA", 0.1, 1.05),
            ("Sa_avg2-Sa_avg2", 0.2, 1.0),
            ("SA-Sa_avg2", 0.2, 1.0),
            ("Sa_avg3-Sa_avg3", 0.2, 1.0),
            ("Sa_avg2-Sa_avg3", 0.2, 1.0),
            ("Sa_avg3-Sa_avg2", 0.2, 1.0),
            ("Sa_avg3-Sa_avg3", 1.0, 1.0),
        ]
    )
    def test_aso2024_ann(self, im_pair, t1, t2):
        val1 = corr.ann_corr(im_pair, t1, t2)
        val2 = corr.aso2024(im_pair, t1, t2)

        if not isinstance(val1, float):
            val1 = val1[0]
        if not isinstance(val2, float):
            val2 = val2[0]

        assert pytest.approx(val1, abs=0.05) == pytest.approx(val2, abs=0.05)
