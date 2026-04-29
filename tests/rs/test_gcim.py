import pytest
from pathlib import Path
import numpy as np

from src.gcim import GCIM
from src.utilities import get_period_im
from src.utilities import export_results

path = Path(__file__).resolve().parent


class TestGCIM:
    metadata = path.parents[0] / "src/assets/NGA_W2_v2.pickle"

    @pytest.mark.parametrize(
        "filename, conditional, data_create", [
            ("input1", True, False),
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

        gcim = GCIM(self.metadata, class_data, conditional=conditional)

        gcim.create(data=create_input)

        if "im-star" in gcim.data:
            imt = gcim.data['im-star']['type']
            imn, _ = get_period_im(imt)

            print(np.exp(gcim.output_create["target"]["mu_lnIMi"][imn]))
            print(gcim.output_create['target']['IMi'][imn])
        gcim.select(data=create_input, output_create=gcim.output_create)

    def test_helpers(self):
        gcim = GCIM(self.metadata)

        gcim.get_supported_rupture_parameters()
        gcim.get_supported_sites_parameters()
        gcim.get_supported_distances_parameters()
        gcim.get_metadata_parameters()
        gcim.available_correlation_models()
        gcim.get_supported_im_component_types()
        gcim.get_available_gsims()

    @pytest.mark.parametrize(
        "id", [
            (1),
            # (2),
            # (3),
            # (4),
            # (5),
            # (6),
            # (7),
            # (8),
            # (9),
        ]
    )
    def test_temp(self, id):
        tag = "conditioned_on_SA(0.5)_no_haz_cons"
        input_file = path / f"applications/savvinos/{tag}/input1_IML{id}.json"

        gcim = GCIM(self.metadata, conditional=True)

        gcim.create(data=input_file)
        import numpy as np
        print(np.exp(gcim.output_create["target"]["mu_lnIMi"]['SA']))
        print(gcim.output_create['target']['IMi']['SA'])
        out = gcim.select(data=input_file, output_create=gcim.output_create)

        export_results(
            path / f"applications/savvinos/{tag}/outs/outs_{id}.json", out, "json")
