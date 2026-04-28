import pytest
from pathlib import Path
import json
import pickle

from djura.fragility_converter.ff import FF
from djura.fragility_converter.ff_approximate import FFApproximate

path = Path(__file__).resolve().parent


@pytest.mark.slow
class TestFF:
    og_input = json.load(open(path / "assets/oq_dis/input.json"))

    @pytest.fixture(scope="function")
    def context(self):
        with open(path / "assets/oq_dis/ctx.pickle", 'rb') as f:
            oq_data = pickle.load(f)

        return oq_data

    @pytest.mark.parametrize(
        "frag1, frag2", [
            ("frags-1.0-0.05.json", "frags-0.5-0.05.json"),
            ("frags-1.0-0.05.json", "frags-1.5-0.05.json"),
            ("frags-0.5-0.05.json", "frags-1.0-0.05.json"),
            ("frags-0.5-0.05.json", "frags-1.5-0.05.json"),
            ("frags-1.5-0.05.json", "frags-0.5-0.05.json"),
            ("frags-1.5-0.05.json", "frags-1.0-0.05.json"),
        ]
    )
    def test_ff(self, context, frag1, frag2, plot=True):
        im1_t = frag1.split("-")[1].replace(".json", "")
        im2_t = frag2.split("-")[1].replace(".json", "")

        frag1 = json.load(open(path / f"assets/frags/{frag1}"))
        frag2 = json.load(open(path / f"assets/frags/{frag2}"))

        im1 = {
            "median": frag1['median'],
            "dispersion": frag1['beta'],
            "name": f"SA({im1_t})",
        }

        print("Original Median and dispersion: "
              f"{frag2['median']:.3f}, {frag2['beta']:.3f}")

        im2 = {
            "name": f"SA({im2_t})",
            "min": 0.001,
            "max": 5.0,
            "num_pts": 100
        }

        ff = FF(im1, im2, data=self.og_input, dis_oq=context)
        im2_probs, im2_range = ff.create()

    def test_approximate(self):
        inputs = json.load(
            open(path / "assets/ff-approximate/approx_SA(0.5).json"))
        ff = json.load(open(path / "assets/ff-approximate/ff.json"))

        imt1, imt2 = "SA(1.0)", "SA(0.5)"
        ds = "1"
        frag1 = ff[imt1][ds]

        im1 = {
            "median": frag1['median'],
            "dispersion": frag1['beta'],
            "name": f"{imt1}",
        }

        im2 = {
            "name": f"{imt2}",
        }
        ff = FFApproximate(
            im1, im2, data=inputs
        )
        im2_probs, im2_range = ff.create()
