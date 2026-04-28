import json
from pathlib import Path

import numpy as np
import pytest

from djura.edp_im.predict import edp_im_batch
from djura.vulnerability_modeller.utilities import to_json_serializable

from conftest import run_edp_im, run_edp_im_infill, run_edp_im_isol


path = Path(__file__).resolve().parent / "assets/backbones"


ec8 = {"period": 0.5, "period_c": "1"}
krawinkler_nassar = {"period": 0.5, "ah": 2}
miranda = {"period": 0.5, "site": 'ROck', "period_g": 1}
newmark_hall = {"period": 0.5, "period_cc": 0.5, "period_c": 1}
vidic = {"period": 0.5, "period_c": 1}
guerrini = {"period": 0.5, "period_c": 0.5, "case": "fd"}
sh = {"damping": 0.075, "ductility": 3, "hardening_ratio": 0.05, "period": 1}


class TestEDPIM:
    @pytest.mark.parametrize(
        "method, hysteresis, im_type, backbone", [
            ("ec8", "BILin", "sa", ec8),
            ("krawinkler_nassar", "bilin", "sa", krawinkler_nassar),
            ("miranda", "bilin", "sa", miranda),
            ("newmark_hall", "bilin", "sa", newmark_hall),
            ("vidic", "bilin", "sa", vidic),
            ("guerrini", "bilin", "sa", guerrini),
            ("shahnazaryan-oreilly", "bilin", "sa", sh),
        ])
    def test_batch(self, method, hysteresis, im_type, backbone):
        result = run_edp_im({
            "method": method,
            "hysteresis": hysteresis,
            "im_type": im_type,
            "backbone": backbone,
        })

        assert result["status"] == "success"

    @pytest.mark.parametrize(
        "filename, method, model", [
            ("backbone1", "backbone", "bilin"),
            ("backbone2", "idealized", "bilin"),
            ("backbone3", "idealized", "bilin"),
            ("backbone4", "spo", "bilin"),
            ("backbone5", "spo", "bilin"),
        ]
    )
    def test_backbone(self, filename, method, model):
        data = json.load(open(path / f"{filename}.json"))

        f_factor = 1
        d_factor = 1

        if method == "spo":
            if filename == "backbone4":
                spo = np.loadtxt(path.parents[0] / "spo/spo_topdisp.txt")
            else:
                # SDOF case
                spo = np.loadtxt(path.parents[0] / "spo/spo_disps.txt")
                f_factor = 800 * 1.3
                d_factor = 1.3

            data["spo"] = {
                "force": spo[:, 0] / f_factor,
                "displacement": spo[:, 1:] / d_factor
            }

        result = run_edp_im({
            "method": 'shahnazaryan-oreilly',
            "hysteresis": model,
            "im_type": 'sa',
            'backboneMethod': method,
            "backbone": to_json_serializable(data),
        })

        assert result["status"] == "success"

    @pytest.mark.parametrize(
        "filename, method, model", [
            ("backbone1", "backbone", "bilin"),
            ("backbone2", "idealized", "bilin"),
            ("backbone3", "idealized", "bilin"),
            ("backbone4", "spo", "bilin"),
            ("backbone5", "spo", "bilin"),
        ]
    )
    def test_sa_avg(self, filename, method, model):
        data = json.load(open(path / f"{filename}.json"))

        f_factor = 1
        d_factor = 1

        if method == "spo":
            if filename == "backbone4":
                spo = np.loadtxt(path.parents[0] / "spo/spo_topdisp.txt")
            else:
                spo = np.loadtxt(path.parents[0] / "spo/spo_topdisp.txt")
                f_factor = 800 * 1.3
                d_factor = 1.3

            data["spo"] = {
                "force": spo[:, 0] / f_factor,
                "displacement": spo[:, 1:] / d_factor
            }

        result = run_edp_im({
            "method": 'shahnazaryan-oreilly',
            "hysteresis": model,
            "im_type": 'sa_avg',
            'backboneMethod': method,
            "backbone": to_json_serializable(data),
        })

        assert result["status"] == "success"

    def test_isol(self):
        data = json.load(open(path / "isol.json"))

        result = run_edp_im_isol(data)

        assert result["status"] == "success"

    @pytest.mark.parametrize(
        "im_type, sdof", [
            ("sa_avg", False),
            ("sa", False),
            ("sa", True),
        ]
    )
    def test_infill(self, im_type, sdof):
        data = json.load(open(path / "infill.json"))

        disp = np.genfromtxt(path.parents[0] / "spo/spo_disps_infill.txt")
        rx = np.genfromtxt(path.parents[0] / "spo/spo_shear_infill.txt")

        spo = {
            "force": rx,
            "displacement": disp
        }

        if sdof:
            spo = {
                "force": rx / 800 / 1.3,
                "displacement": disp / 1.3
            }

        body = {
            "backbone": {
                "spo": to_json_serializable(spo),
                "damping": data["damping"],
                "gamma": data["gamma"],
                "mstar": data["mstar"],
                "sdof": sdof
            },
            "im_type": im_type,
            "backboneMethod": "spo",
        }

        result = run_edp_im_infill(body)

        assert result["status"] == "success"

    def test_cases_batch(self):
        data = json.load(open(path / "batch.json"))

        result = edp_im_batch(data)

        assert result["status"] == "success"
