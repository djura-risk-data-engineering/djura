"""Exact (OpenQuake disaggregation) route with two GMMs per source group"""
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from djura.record_selection.gcim import GCIM

path = Path(__file__).resolve().parents[1] / "ff" / "assets" / "oq_dis"

IM_REF = "SA(0.2)"
LT_WEIGHTS = np.array([0.3, 0.2, 0.35, 0.15])


@pytest.fixture(scope="module")
def dis_oq():
    with open(path / "ctx.pickle", "rb") as f:
        ctx = pickle.load(f)
    ctx = deepcopy(ctx)
    ctx["im_ref"] = IM_REF
    ctx["lt-weights"] = LT_WEIGHTS
    (grp, data), = ctx["data"].items()
    data["gsims"] = [
        {"AristeidouEtAl2024": np.array([0, 2], dtype=np.uint32)},
        {"BooreEtAl2014": np.array([1, 3], dtype=np.uint32)},
    ]
    data["parameters"] = [{}, {}]
    return ctx


def _gcim(dis_oq):
    poe = dis_oq["oq-poes"][4]
    return GCIM({}, conditional=True, dis_oq=deepcopy(dis_oq),
                poe_for_selection=poe)


def test_group_required_parameters(dis_oq):
    ctx = deepcopy(dis_oq)
    (grp, data), = ctx["data"].items()
    data["required-parameters"] = {"mag", "rake", "rjb", "rrup", "ztor"}
    ctx["required-parameters"] = {"mag"}  # global set is not used
    rows = ctx["ctx_by_grp"][grp]
    rows["ztor"][0] = np.nan

    ruptures = GCIM({}, conditional=True, dis_oq=ctx,
                    poe_for_selection=ctx["oq-poes"][4]).data["ruptures"]

    assert "ztor" not in ruptures[0]
    assert ruptures[1]["ztor"] == rows["ztor"][1]
    assert {"mag", "rake", "rjb", "rrup"} <= set(ruptures[1])
    assert "rx" not in ruptures[1] and "hypo_depth" not in ruptures[1]


def test_gmm_weights_sum_over_realizations(dis_oq):
    gcim = _gcim(dis_oq)

    for group in gcim.data["gmms"]:
        (im, gmm), = group.items()
        assert gmm["names"] == ["AristeidouEtAl2024", "BooreEtAl2014"]
        np.testing.assert_allclose(gmm["weights"], [0.65, 0.35])


@pytest.mark.slow
def test_exact_target(dis_oq):
    gcim = _gcim(dis_oq)
    out = gcim.create()

    weights = out["data"]["weights_imi"]["SA"]
    assert np.sum(weights) == pytest.approx(1.0)

    # Each scenario carries the weight of its own rupture and GMM
    scenarios = gcim.get_all_scenarios()["SA"]
    ws = out["data"]["eps_weights"]
    position, expected = {}, []
    for sc in scenarios:
        k = position.get(sc["gmm"], 0)
        position[sc["gmm"]] = k + 1
        expected.append(ws[sc["gmm"]][k])
    expected = np.asarray(expected, dtype=float).squeeze()
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert np.count_nonzero(weights) > len(weights) // 2

    # The target passes through IM*
    target = out["target"]
    i = list(target["IMi"]["SA"]).index(0.2)
    im_star = gcim.data["im-star"]["value"]
    assert np.exp(target["mu_lnIMi"]["SA"][i]) == pytest.approx(im_star)
    assert target["sigma_lnIMi"]["SA"][i] == pytest.approx(0, abs=1e-6)
