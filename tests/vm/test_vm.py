import copy
import json
from pathlib import Path

import pytest

from djura.edp_im.predict import edp_im_batch
from djura.vulnerability_modeller.utilities import deep_merge

from conftest import run_acceleration, run_drift, run_vm


path = Path(__file__).resolve().parent / "assets"


data_edp_im = [
    {
        "hysteresis": "bilin",
        "im_type": "sa",
        "method": "shahnazaryan-oreilly",
        "backboneMethod": "backbone",
        "backbone": {
            "damping": 0.02,
            "ductility": 3,
            "ductility_f": 6,
            "hardening_ratio": 0.05,
            "period": 1
        }
    },
    {
        "hysteresis": "bilin",
        "im_type": "sa",
        "method": "shahnazaryan-oreilly",
        "backboneMethod": "backbone",
        "backbone": {
            "damping": 0.03,
            "ductility": 3,
            "ductility_f": 6,
            "hardening_ratio": 0.02,
            "period": 0.5
        }
    },
    {
        "hysteresis": "infill",
        "im_type": "sa",
        "backboneMethod": "backbone",
        "backbone": {
            "period": 1,
            "c_y": 3.5,
            "c_rp": 2,
            "mu_h": 3.2,
            "mu_s": 3.2,
            "mu_rp": 3.5,
            "mu_ult": 10
        }
    },
]


data_est = [
    {
        "backbone": {
            "heights": [3.5, 3.0, 3.0, 3.0, 3.0],
            "dy": 0.03,
            "ductility_f": 6,
        },
        "building-type": "frame"
    },
    {
        "backbone": {
            "heights": [4.5, 3.5, 3.5],
            "dy": 0.035,
            "ductility_f": 6,
        },
        "building-type": "frame"
    },
    {
        "backbone": {
            "heights": [3.5, 3.0, 3.0, 3.0, 3.0],
            "dy": 0.02,
            "ductility_f": 6,
        },
        "building-type": "infill"
    },
]


hazard = {
    "pga": [0.0001144, 2.6348751, 0.2340722],
    "im": [0.000298042, 2.551209397, 0.242236676]
}


class TestVM:
    slfs = json.load(open(path / "vm/slf.json"))
    rc = 400000

    @pytest.fixture()
    def edp_ims(self):
        return edp_im_batch(data_edp_im)

    @pytest.fixture()
    def psds(self):
        return run_drift(data_est)

    @pytest.fixture()
    def pfas(self, edp_ims, psds):
        data = [deep_merge(d1, d2) for d1, d2 in zip(data_est, data_edp_im)]

        for i, _data in enumerate(data):
            _data["pfa-model"] = "muho"
            _data["hazard"] = hazard

            if i in psds["success_ids"]:
                _data["psd"] = next((r for r in psds["successes"]
                                     if r["index"] == i), None)["out"]["psd"]
            else:
                _data["psd"] = []

            if i in edp_ims["success_ids"]:
                _data["sr"] = next((r for r in edp_ims["successes"]
                                    if r["index"] == i), None
                                   )["out"]["strength-ratios"]
            else:
                _data["sr"] = []

        return run_acceleration(data)

    def test_run(self, edp_ims, psds, pfas):
        n = len(edp_ims["failure_ids"]) + len(edp_ims["success_ids"])

        slfs = [copy.deepcopy(self.slfs) for _ in range(n)]
        slfs[0]["1 - NS: PFA"]["Storey"] = [0, 1, 2, 3, 4, 5]
        slfs[0]["3 - NS: PSD"]["Storey"] = [1, 2, 3, 4, 5]
        slfs[0]["2 - S: PSD"]["Storey"] = [1, 2, 3, 4, 5]
        slfs[1]["1 - NS: PFA"]["Storey"] = [0, 1, 2, 3]
        slfs[1]["3 - NS: PSD"]["Storey"] = [1, 2, 3]
        slfs[1]["2 - S: PSD"]["Storey"] = [1, 2, 3]
        # The third one does not matter, as it should fail

        for slf in slfs:
            slf["rc"] = self.rc

        slfs[0]["cases"] = [0]
        slfs[1]["cases"] = [1]
        slfs[2]["cases"] = [2]

        data = {
            "slfs": slfs,
            "edp-ims": edp_ims,
            "psds": psds,
            "pfas": pfas
        }

        result = run_vm(data)

        assert result["status"] == "success"
