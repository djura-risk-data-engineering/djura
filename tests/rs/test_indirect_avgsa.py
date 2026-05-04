"""
DEV:
To run the test locally

```sh
export DJURA_METADATA_PATH=src/djura/record_selection/assets/NGA_W2_v2.pickle
pytest -m slow
```

PowerShell (Windows):

```powershell
$env:DJURA_METADATA_PATH = `
    "src/djura/record_selection/assets/NGA_W2_v2.pickle"
pytest -m slow
```

CMD (Windows):

```cmd
set DJURA_METADATA_PATH=src/djura/record_selection/assets/NGA_W2_v2.pickle
pytest -m slow
```
"""

import pytest
from pathlib import Path

from djura.record_selection.gcim import GCIM

path = Path(__file__).resolve().parent


@pytest.mark.slow
class TestIndirectAvgSa:

    @pytest.mark.parametrize(
        "filename", [
            "input13"
        ]
    )
    def test_gsim(self, filename):
        input_file = path / f"assets/gcim_inputs/{filename}.json"

        gcim = GCIM(input_file, conditional=True)

        gcim.create()
        import numpy as np
        print(np.exp(gcim.output_create["target"]["mu_lnIMi"]['Sa_avg2']))
        print(gcim.output_create['target']['IMi']['Sa_avg2'])
