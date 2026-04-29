import pytest
from pathlib import Path

from src.gcim import GCIM

path = Path(__file__).resolve().parent


class TestIndirectAvgSa:
    metadata = path.parents[0] / "src/assets/NGA_W2_v2.pickle"

    @pytest.mark.parametrize(
        "filename", [
            "input13"
        ]
    )
    def test_gsim(self, filename):
        input_file = path / f"assets/gcim_inputs/{filename}.json"

        gcim = GCIM(self.metadata, input_file, conditional=True)

        gcim.create()
        import numpy as np
        print(np.exp(gcim.output_create["target"]["mu_lnIMi"]['Sa_avg2']))
        print(gcim.output_create['target']['IMi']['Sa_avg2'])
