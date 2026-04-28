import json
from pathlib import Path

import pytest

from djura.edp_im.predict import edp_im_batch
from djura.vulnerability_modeller.utilities import deep_merge

from conftest import run_acceleration, run_drift, run_vm


path = Path(__file__).resolve().parent / "assets/slf-refined"


class TestSLFRefined:
    """
    Tests for SLF refined paper:
    O'Reilly, G. J., & Shahnazaryan, D. (2024). On the utility of story loss
    functions for regional seismic vulnerability modeling and risk
    assessment. Earthquake Spectra, 40(3), 1933-1955.
    https://doi.org/10.1177/87552930241245940
    """
    @pytest.fixture(params=[
        500000,     # Normalize by a smaller cost
        1105000,    # Normalize by total cost of all components
        2000000,    # Normalize by a bigger cost
    ])
    def slf(self, request):
        rc = request.param
        slfs = json.load(open(path / "slf.json"))
        slfs[0]["rc"] = rc
        return slfs

    @pytest.fixture()
    def edp_im(self):
        data = [{
            "method": "shahnazaryan-oreilly",
            "hysteresis": "bilin",
            "im_type": "sa_avg",
            "backboneMethod": "backbone",
            "backbone": {
                "damping": 0.05,
                "ductility": 4,
                "ductility_f": 12,
                "hardening_ratio": 0.02,
                "period": 1
            },
        }]

        return data, edp_im_batch(data)

    @pytest.fixture()
    def psd(self):
        data = [
            {
                "backbone": {
                    "heights": [3.5, 3.0, 3.0, 3.0],
                    "dy": 0.04969804057656669,
                    "ductility_f": 12,
                },
                "building-type": "frame"
            },
        ]
        return data, run_drift(data)

    @pytest.fixture()
    def pfa(self, psd, edp_im):
        hazard = {
            "pga": [0.0001144, 2.6348751, 0.2340722],
            "im": [3.309e-5, 2.812677, 0.233812]
        }

        data = [deep_merge(d1, d2) for d1, d2 in
                zip(psd[0], edp_im[0])]

        for i, _data in enumerate(data):
            _data["pfa-model"] = "muho"
            _data["hazard"] = hazard

            if i in psd[1]["success_ids"]:
                _data["psd"] = next((r for r in psd[1]["successes"]
                                     if r["index"] == i), None)["out"]["psd"]
            else:
                _data["psd"] = []

            if i in edp_im[1]["success_ids"]:
                _data["sr"] = next((r for r in edp_im[1]["successes"]
                                    if r["index"] == i), None
                                   )["out"]["strength-ratios"]
            else:
                _data["sr"] = []

        return run_acceleration(data)

    @pytest.fixture()
    def vm(self, slf, edp_im, psd, pfa):
        psd = psd[1]
        edp_im = edp_im[1]

        slf[0]["demolition"] = {
            "median": 1.0,
            "beta": 0.3,
        } if slf[0]["rc"] != 2000000 else None

        data = {
            "slfs": slf,
            "edp-ims": edp_im,
            "psds": psd,
            "pfas": pfa,
        }

        return run_vm(data)

    def test_run(self, vm):
        assert vm["status"] == "success"
