import pytest
from pathlib import Path
import json

from src.gcim import GCIM
from src.code_select import CodeSelect

path = Path(__file__).resolve().parent


class TestIgnoreByKey:
    metadata = path.parents[0] / "src/assets/NGA_W2_v2.pickle"

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

        gcim = GCIM(self.metadata, class_data, conditional=conditional)

        gcim.create()
        gcim.select(output_create=gcim.output_create)

    @pytest.fixture(scope="function", params=[
        (11, 2, 0),
    ])
    def code(self, request):
        num_records = request.param[0]
        num_components = request.param[1]
        greedy_loops = request.param[2]
        context_limits = {
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

        code = CodeSelect(
            self.metadata, num_records=num_records,
            num_components=num_components, greedy_loops=greedy_loops,
            context_limits=context_limits,
        )
        return code

    @pytest.mark.parametrize(
        "period, bounds, ag, imp_class, spectrum_type, soil, vs30, s, tb, tc, td, ag_ratio, tb_v, tc_v, td_v, st, topography, loose_surface_layer, sf_lim, sf_lim_v, bounds_v, target", [
            (1.0, [None, None], 0.2, 'II', "Type-1", "A", None, None, None, None, None, None,
             None, None, None, None, 'T1', False, [0.5, 2.0], [None, None], [None, None], False),
        ]
    )
    def test_code(self, code: CodeSelect, period, bounds, ag, imp_class,
                  spectrum_type, soil, vs30, s, tb, tc, td, ag_ratio,
                  tb_v, tc_v, td_v, st, topography, loose_surface_layer,
                  sf_lim, sf_lim_v, bounds_v, target):

        code._create_cen2004(
            ag, imp_class, spectrum_type, soil, vs30, s, tb, tc, td, ag_ratio, tb_v, tc_v, td_v, st, topography, loose_surface_layer
        )

        if not target:
            code.output_create['target'] = None

        code.select_cen2004(
            period, ag, imp_class, spectrum_type, soil, vs30,
            s, tb, tc, td, ag_ratio, tb_v, tc_v, td_v,
            st, topography, loose_surface_layer,
            sf_lim[0], sf_lim[1], bounds[0], bounds[1],
            sf_lim_v[0], sf_lim_v[1], bounds_v[0], bounds_v[1]
        )
