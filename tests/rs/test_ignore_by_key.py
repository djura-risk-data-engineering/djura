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
import json

from djura.record_selection.gcim import GCIM

path = Path(__file__).resolve().parent


@pytest.mark.slow
class TestIgnoreByKey:
    """
    Tests support to ignore earthquake records by key (text string)

    'EQ_name' in 'context_limits'
    """

    @pytest.mark.parametrize(
        "filename, conditional", [
            ("input1", True),
        ]
    )
    def test_gcim(self, filename, conditional):
        input_file = path / f"assets/gcim_inputs/{filename}.json"
        class_data = json.load(open(input_file))
        class_data['context_limits'] = {
            'EQ_name': "Helena, ;",
            "magnitude": [6, 7],
            "mechanism": [
                {
                    "text": "normal fault",
                    "value": 1
                },
                {
                    "text": "reverse fault",
                    "value": 2
                }
            ],
        }

        gcim = GCIM(class_data, conditional=conditional)

        gcim.create()
        gcim.select(output_create=gcim.output_create)
