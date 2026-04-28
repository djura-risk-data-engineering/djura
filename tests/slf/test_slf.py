import json
import pytest
from pathlib import Path

from djura.slf.slf import SLF
from djura.slf.utilities import filter_args

path = Path(__file__).resolve().parent


class TestSLF:

    @pytest.mark.parametrize(
        "filename, filecorr, corr, grouping, regression, seed",
        [
            ('inv_pfa_psd', None, False, False, "weibull", 0),
            ('inv_pfa_psd', None, False, True, "weibull", 0),
            ('inventory', 'correlations', False, True, "weibull", 0),
            ('inventory', 'correlations', False, False, "papadopoulos", 1),
            ('pfa_inv', 'pfa_corr', False, False, "weibull", 1),
            ('inv1', 'corr1', False, False, "weibull", 0),
            ('inv2', None, True, False, "weibull", 0),
        ],
    )
    def test_slf(self, filename, filecorr, corr, grouping,
                 regression, seed):
        correlations = None
        if filecorr is not None:
            correlations = json.load(open(path / f"assets/{filecorr}.json"))
        inventory = json.load(open(path / f"assets/{filename}.json"))

        data = {
            "inventory": inventory,
            "correlations": correlations,
            "do_grouping": grouping,
            "include_correlations": corr,
            "conversion": '1.0',
            "realizations": '20',
            "replacement_cost": '1.0',
            "regression": regression,
            "storey": None,
            "directionality": None,
            "seed": str(seed),
        }

        include_corr = data.get('include_correlations', False)
        if not include_corr:
            data['correlations'] = None

        inventory = data['inventory']
        correlations = data['correlations']
        unique_edp = set(item['EDP'] for item in inventory)

        outs = {}
        n_prev = 0
        for edp in unique_edp:
            data["edp"] = edp.lower()
            data['inventory'] = [item for item in inventory
                                 if item['EDP'] == edp]

            id_map = {old_id: new_id for new_id, old_id in enumerate(
                sorted(item['id'] for item in data['inventory']))}
            for item in data['inventory']:
                item['id'] = id_map[item['id']]

            if correlations is not None:
                data['correlations'] = [
                    correlations[i] for (i, item)
                    in enumerate(data['inventory'])
                    if item['EDP'] == edp
                ]

            print(f"{edp}: {len(data['inventory'])} elements have been"
                  f" processed; Include correlations: {include_corr}")

            filtered_data = filter_args(SLF, data)
            filtered_data['n_prev'] = n_prev

            slf_obj = SLF(**filtered_data)
            out = slf_obj.generate_slfs()

            outs.update(out)
            del slf_obj
