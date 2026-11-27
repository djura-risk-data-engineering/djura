import json
import numpy as np
import pytest
from pathlib import Path
from scipy.stats import norm

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


def _item(median, beta, cost, cost_dispersion=0.1):
    return [{
        "id": 0, "name": "0", "EDP": "PSD", "Component": "NS", "Group": "",
        "Quantity": 1, "damage-states": 1, "median-demand": [median],
        "total-dispersion": [beta], "repair-cost": [cost],
        "cost-dispersion": [cost_dispersion], "best-fit": [None],
    }]


def test_fragility_is_lognormal_median_dispersion():
    median, beta = 0.01, 0.5
    edps = np.array([median, median * np.exp(beta)])
    slf = SLF(_item(median, beta, 1.0), "PSD", edp_range=edps)
    frag = slf.fragility_function()["ITEMs"][0][:, 0]
    np.testing.assert_allclose(frag, [0.5, norm.cdf(1.0)])


def test_costs_not_truncated():
    slf = SLF(_item(1e-4, 0.01, 1.5, 0.01), "PSD", edp_range=np.array([1.0]),
              realizations=50, seed=1)
    damage = slf.perform_monte_carlo(slf.fragility_function())
    _, _, costs = slf.calculate_costs(damage)
    sampled = np.array([costs[0][n][0] for n in range(50)])
    assert np.all(np.abs(sampled - 1.5) < 0.1)


def test_mismatched_ds_lengths_raise():
    item = _item(0.01, 0.5, 1.0)
    item[0]["cost-dispersion"] = [0.1, 0.1]
    with pytest.raises(ValueError):
        SLF(item, "PSD")


def test_seed_zero_reproducible():
    def run():
        slf = SLF(_item(0.01, 0.5, 1.0), "PSD", edp_range=np.array([0.01]),
                  realizations=200, seed=0)
        damage = slf.perform_monte_carlo(slf.fragility_function())
        return np.array([damage[0][n][0] for n in range(200)])

    np.random.seed(123)
    first = run()
    np.random.seed(456)
    np.testing.assert_array_equal(first, run())
