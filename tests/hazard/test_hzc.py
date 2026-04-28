from djura.record_selection.utilities import (
    proc_oq_hazard_curve, get_rs_imi_intensities,
)
from djura.hazard_consistency.hazard_consistency import HazardConsistency
import pytest
from pathlib import Path

path = Path(__file__).resolve().parent
data_path = path / "assets"


class TestHZC:
    imts = [
        "PGA", "FIV3(1.0)", "Sa_avg2(0.5)", "SA(1.5)", "PGV"
    ]
    IM_START_ACC = 0.01
    IM_END_ACC = 8.0
    IM_START_VEL = 1.0
    IM_END_VEL = 300.0
    NUM_IM = 500

    @pytest.fixture
    def hz(self):
        hz = proc_oq_hazard_curve(
            [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005,
             0.0025, 0.001, 0.0004, 0.0002, 0.0001],
            data_path / "hazard",
            json_file=data_path / "hazard/hazard.json"
        )
        return hz

    def get_hzc_results(self, hz, imt):
        cond_imls = hz["cond_imls"][imt]
        cond_poes = hz["cond_poes"]
        inv_t = hz["investigation_time"]

        model = HazardConsistency(
            conditional_intensities=cond_imls,
            conditional_poes=cond_poes,
            investigation_time=inv_t
        )

        outs = {}
        for imi in self.imts:
            imls = get_rs_imi_intensities(
                selection_dir=data_path / f"records/{imt}",
                poes=cond_poes,
                imi=imi
            )

            outs[imi] = model.check(
                imls, num_im=self.NUM_IM
            )
        return outs

    @pytest.mark.parametrize("imt", imts)
    def test_function(self, hz, imt):
        print("Running for IM:", imt)

        self.get_hzc_results(hz, imt)
