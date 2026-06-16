# flake8: noqa

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
class TestGCIMV:
    def test_gcim_v(self):
        input_file = path / "assets/gcim_inputs/input17.json"

        gcim = GCIM(input_file, conditional=True)
        outs = gcim.create()
        rs = gcim.select(output_create=gcim.output_create)

        # from djura.utilities import export_results
        # export_results("temp", outs, "json")
        # export_results("temp_rs", rs, "json")
