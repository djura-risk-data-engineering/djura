# flake8: noqa
import pytest
from pathlib import Path
import json
import pickle
from numpy import exp

from src.utilities import export_results
from src.gcim import GCIM

path = Path(__file__).resolve().parent


class TestOQRS:
    metadata = path.parents[0] / "src/assets/NGA_W2_v2.pickle"

    """
    Reads preprocessed OQ disaggregation data and performs record selection 
    using the generated context information. 

    If input data is provided for Djura record selection,
    it will be overridden by the OQ disaggregation data.
    """

    @pytest.mark.parametrize(
        "filename, imt, n_rups, n_leafs, im_star", [
            ("ctx_80_AvgSA(0.5)", "Sa_avg", 798, 2, 0.38),
        ]
    )
    def test_oq_dis_select(self, filename, imt, n_rups, n_leafs, im_star):
        og_input = json.load(open(path / "assets/oq_dis/input.json"))

        with open(path / f"assets/oq_dis/{filename}.pickle", 'rb') as f:
            oq_data = pickle.load(f)

        gcim = GCIM(self.metadata, og_input, conditional=True,
                    dis_oq=oq_data, poe_for_selection=0.1)

        # Number of logic tree leafs for each IM (SA for example)
        # N_logic_tree_leafs = n_gmms * n_sources
        w_tot, leafs = 0, 0
        for gmm in gcim.data['gmms']:
            im_key = next(iter(gmm))
            if im_key == imt:
                w_tot += sum(gmm[imt]['weights'])
                leafs += len(gmm[imt]['names'])

        assert leafs == n_leafs

        # Sum of logic tree leaf weights
        assert w_tot == pytest.approx(1.0, abs=0.01)

        # Total number of ruptures (all sources)
        assert len(gcim.data['ruptures']) == n_rups

        # Value of IM*
        assert pytest.approx(
            gcim.data['im-star']['value'], abs=0.01) == im_star

        # ----------- Create target
        gcim.create()

        # Validate outputs, target IM*
        im_star_period = 0.5
        idx_im_star = gcim.output_create['target']['IMi']['Sa_avg2'].index(
            im_star_period)
        im_star_value = exp(
            gcim.output_create['target']['mu_lnIMi']['Sa_avg2'][idx_im_star])

        assert pytest.approx(im_star, abs=0.1) == im_star_value

    def test_ctx(self):
        og_input = json.load(open(path / "assets/oq_dis/input.json"))

        with open(path / "assets/oq_dis/ctx.pickle", 'rb') as f:
            oq_data = pickle.load(f)

        oq_data['im_ref'] = "SA(0.5)"

        gcim = GCIM(self.metadata, og_input, conditional=True,
                    dis_oq=oq_data, poe_for_selection=0.0025)
        gcim.create()
