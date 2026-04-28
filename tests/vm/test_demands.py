from pathlib import Path

import pytest

from djura.edp_im.predict import edp_im_batch
from djura.vulnerability_modeller.utilities import deep_merge

from conftest import run_acceleration, run_drift


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


class TestDemands:
    @pytest.fixture()
    def edp_ims(self):
        return edp_im_batch(data_edp_im)

    @pytest.fixture(scope="function")
    def drift_response(self):
        return run_drift(data_est)

    def test_drifts(self, drift_response):
        assert drift_response is not None
        assert drift_response["status"] == "success"

    def test_acc(self, edp_ims, drift_response):
        data = [deep_merge(d1, d2) for d1, d2 in zip(data_est, data_edp_im)]

        for i, _data in enumerate(data):
            _data["pfa-model"] = "muho"
            _data["hazard"] = hazard

            if i in drift_response["success_ids"]:
                _data["psd"] = next((r for r in drift_response["successes"]
                                     if r["index"] == i), None)["out"]["psd"]
            else:
                _data["psd"] = []

            if i in edp_ims["success_ids"]:
                _data["sr"] = next((r for r in edp_ims["successes"]
                                    if r["index"] == i), None
                                   )["out"]["strength-ratios"]
            else:
                _data["sr"] = []

        result = run_acceleration(data)

        assert result["status"] == "success"
