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
class TestGCIM:

    @pytest.mark.parametrize(
        "filename, conditional, data_create", [
            ("input1", True, False),
            ("input2", None, False),
            ("input3", None, False),
            ("input1", None, True),
            ("input4", True, False),
            ("input5", True, False),
            ("input6", True, False),
            ("input7", True, False),
            ("input8", True, False),
            ("input9", True, False),
            ("input10", True, False),
            ("input11", True, False),
            ("input12", True, False),
            ("input14", True, False),
            ("input15", True, False),
            ("input16", True, False),
            ("input18", True, True),
        ]
    )
    def test_gcim(self, filename, conditional, data_create):
        input_file = path / f"assets/gcim_inputs/{filename}.json"
        class_data = None
        create_input = None

        if data_create:
            create_input = input_file
        else:
            class_data = input_file

        gcim = GCIM(class_data, conditional=conditional)

        gcim.create(data=create_input)
        gcim.select(data=create_input, output_create=gcim.output_create)

    def test_helpers(self):
        gcim = GCIM()

        gcim.get_supported_rupture_parameters()
        gcim.get_supported_sites_parameters()
        gcim.get_supported_distances_parameters()
        gcim.get_metadata_parameters()
        gcim.available_correlation_models()
        gcim.get_supported_im_component_types()
        gcim.get_available_gsims()
