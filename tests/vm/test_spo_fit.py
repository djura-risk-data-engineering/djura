from pathlib import Path
import numpy as np
from djura.vulnerability_modeller.backbone import Backbone


path = Path(__file__).resolve().parent / "assets/backbones"


class TestSPOFit:
    def test_spo(self):
        spo = np.loadtxt(path.parents[0] / "spo/spo_topdisp.txt")

        data = {
            "spo": {
                "force": spo[:, 0],
                "displacement": spo[:, 1:].reshape(-1)
            }
        }

        b = Backbone(data, "spo", "bilin")
        print(b.backbone)
