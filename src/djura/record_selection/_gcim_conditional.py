from typing import List
from numpy import zeros, log

from .gmm_tools import calculate_epsilon, get_mean_xy, get_sigma_xy
from ._gcim import _GCIM
from ._gcim_select import _GCIMSelect


class _GCIMConditional(_GCIM, _GCIMSelect):

    def __init__(self, metadata, use_disaggregation: bool) -> None:

        self.realization = None
        self.selected_scaled_total: dict = {}
        self.selected_scaled_best: dict = {}
        self.output_create: dict = {
            "target": None,
            "num_components": None,
            "component_definition": None,
        }

        _GCIM.__init__(self, metadata, use_disaggregation)
        _GCIMSelect.__init__(self, self.output_create, metadata)

    def create(
        self,
        im_star: dict,
        gmms: dict,
        ruptures: dict,
        total_weights: dict,
        imi: List[str],
        num_components: int,
        component_definition: str,
        add_data_for_dis: dict = None,
        avg_sa: dict = None,
    ):
        """Creates conditional GCIM distribution

        Parameters
        ----------
        im_star : dict
            Conditional IM
        gmms : dict
            GMMs and associated weights
        ruptures : dict
            Rupture scenarios and associated weights and context parameters
        total_weights : dict
            Total weights of all scenarios for each IMi
            If we have 2 ruptures, 1st rupture has 3 GMMs, 2nd rupture has
            2 GMMs the order of index shall go
            [rup1gmm1, rup1gmm2, rup1gmm3, rup2gmm1, rup2gmm2]
        imi : List[str]
            IMi of interest
        num_components : int
            Number of components to consider
        component_definition : str
            Component definition
        """
        num_components = int(num_components)

        self.output_create["num_components"] = num_components
        if num_components == 1:
            component_definition = "RotD50"

        self.output_create["component_definition"] = component_definition

        if avg_sa is not None:
            im_star, imi, gmms, total_weights = self._identify_indirect_sa_avg(
                avg_sa, im_star, imi, gmms, total_weights
            )

        scenarios, imi, mu_rup, sigma_rup, cov_rup, gmms_dict = \
            self._initialize_create(ruptures, imi, gmms, total_weights)

        mu_imstar_rup = {}
        sigma_imstar_rup = {}
        cov_imstar_rup = {}
        for _im, arr in mu_rup.items():
            mu_imstar_rup[_im] = zeros(arr.shape)
            sigma_imstar_rup[_im] = zeros(sigma_rup[_im].shape)
            cov_imstar_rup[_im] = zeros(cov_rup[_im].shape)

        # Compute correlation coefficients for IM* and IMis of interest
        # \rho_{lnIMi,lnIM*|Rup}
        rho_imi_im_star = self._get_im_star_imi_correlations(im_star, imi)

        # Conditioned on IM*, using the gmm for the IM*
        # and the same rupture context
        # mu_im_star = zeros((len(ruptures), len(im_star["gmms"]["names"])))
        # sigma_im_star = zeros(
        #     (len(ruptures), len(im_star["gmms"]["names"])))
        # weights_im_star = zeros(
        #     (len(ruptures), len(im_star["gmms"]["names"])))

        # Conditioned on IM*, {rup_i: [gmm1, gmm2, ..., gmmn]}
        mu_im_star = {}
        sigma_im_star = {}
        eps_im_star = {}
        eps_im_star_comb = {}
        weights_im_star = {}
        for i, _rupture in enumerate(ruptures):
            mu_im_star[i], sigma_im_star[i], weights_im_star[i] = \
                self.get_conditional_im(
                im_star, _rupture, gmms, component_definition,
                num_components)

        # Back-calculate epsilon at the IM*
        # Shape = (len(ruptures), len(im_star_gmms))
        # keys: GMM_name(s) and 'ws' in case of disaggregation
        im_star_gmms = set()
        eps_im_star['ws'] = {}
        for i, _mu_im_star in mu_im_star.items():
            gmm_ids = ruptures[i]['gmms']
            rup_gmms = self._find_gmm_by_key(gmms, im_star['type'], gmm_ids)

            im_star_gmms = im_star_gmms.union(set(_mu_im_star.keys()))
            eps_im_star[i] = {}
            for g, _mu in _mu_im_star.items():
                eps_im_star[i][g] = calculate_epsilon(
                    im_star["value"], _mu, sigma_im_star[i][g])
                if self.USE_FULL_PSHA:
                    _g_idx = rup_gmms['names'].index(g)

                    # Logic tree leaf weight
                    lt_weight = rup_gmms['weights'][_g_idx]

                    _poe = self._truncnorm_sf(
                        add_data_for_dis['phi_b'], eps_im_star[i][g]
                    )
                    ws = -log((1 - ruptures[i]['probs']) ** _poe) / \
                        add_data_for_dis['inv_time']
                    ws /= -log(1 - add_data_for_dis['poe_for_selection']) / \
                        add_data_for_dis['inv_time']

                    if g in eps_im_star['ws']:
                        eps_im_star['ws'][g].append(ws * lt_weight)
                    else:
                        eps_im_star['ws'][g] = [ws * lt_weight]

        # Used in the case where GMM does not apply to IM*
        # Not sure about this... this is an approximation for convenience
        # No reference for it
        # In the case of usage of diassgregation results,
        # this will never be used
        for i, _rupture in enumerate(ruptures):
            eps_im_star_comb[i] = self._calculate_epsilon_for_rup(
                im_star["value"], mu_im_star[i], sigma_im_star[i],
                weights_im_star[i]
            )

        for im, _scenarios in scenarios.items():
            for i, scenario in enumerate(_scenarios):

                # Retrieve epsilon
                if scenario["gmm"] in im_star_gmms:
                    epsilon = eps_im_star[scenario["rup_id"]][scenario["gmm"]]
                else:
                    epsilon = eps_im_star_comb[scenario["rup_id"]]

                scenario["im_name"] = im

                # Estimate means and stddevs conditioned
                # on rupture scenario, \mu_{ln(IMi|rup)}, \sigma_{ln(IMi|rup)}
                mu_rup[im][:, i], sigma_rup[im][:, i] = \
                    self._get_all_means_stds(
                        imi[im], scenario, component_definition,
                        num_components)

                # Compute \sigma_{ln(IMi|IM*,rup)}
                sigma_imstar_rup[im][:, i] = get_sigma_xy(
                    sigma_rup[im][:, i], rho_imi_im_star[im])

                # Compute \mu_{ln(IMi|IM*,rup)}
                # Initially compute for each GMM case of IM*
                # shape=(len(gmms_im*), len(imi[im]))
                mu_imstar_rup[im][:, i] = get_mean_xy(
                    mu_rup[im][:, i], sigma_rup[im][:, i], rho_imi_im_star[im],
                    epsilon)

        # Logarithmic median of target spectrum accounting for
        # all cases (GMMs and rupture scenarios) through contribution factors
        # Exact approach
        # Initialize target mean distribution \mu_{lnIMi}
        mu_exact = {}
        weights_imi = {}
        for im_type, means in mu_imstar_rup.items():
            if self.USE_FULL_PSHA:
                # Weights from occurrence rates of ruptures
                # For a single GMM
                weights = self._get_weight_disaggregation(
                    eps_im_star, gmms_dict[im_type])

            else:
                # Logic tree weights
                weights = self._get_par_from_scenarios("w", scenarios[im_type])
            weights_imi[im_type] = weights

            mu_exact[im_type] = self.calc_exact_mean(means, weights)

        # Compute correlation matrix for each IMi
        corr, _corr, _im_idxs = self._get_imi_correlation_matrix(imi)

        # Compute correlation matrix conditioned on IM*
        corr_cond, _corr_cond = self._get_conditional_correlation(
            _corr, rho_imi_im_star, _im_idxs)

        # COV_{ln(IMi|rup)}
        cov_imstar_rup = self._get_all_imi_cov_matrices(
            corr_cond, sigma_imstar_rup, cov_imstar_rup
        )

        # \sigma_{IMi|rup}, COV_{IMi|rup}
        sigma_exact = {}
        cov_exact = {}
        for im_type, means in mu_imstar_rup.items():
            sigma_exact[im_type], cov_exact[im_type] = \
                self.calc_exact_sigma_cov(
                means, cov_imstar_rup[im_type], mu_exact[im_type],
                weights_imi[im_type]
            )

        # Collect the output
        self.output_create["im-star"] = im_star
        self.output_create["corr_imi_imj"] = rho_imi_im_star

        target = {
            "mu_lnIMi": mu_exact,
            "sigma_lnIMi": sigma_exact,
            "cov_lnIMi": cov_exact,
            "IMi": imi,
            "correlations": corr_cond,
            "combined_correlations": _corr_cond,
            "im_idxs": _im_idxs,
        }
        self.output_create["target"] = target

        eps_weights = eps_im_star.pop('ws')

        # Intermediate results
        self.output_create["data"] = {
            "mu_lnIMi_rup": mu_rup,
            "sigma_lnIMi_rup": sigma_rup,
            "cov_lnIMi_rup": cov_rup,
            "mu_lnIMj_rup": mu_im_star,
            "sigma_lnIMj_rup": sigma_im_star,
            "epsilon_lnIMj_rup": eps_im_star,
            "eps_weights": eps_weights,
            "mu_lnIMi_lnIMj_rup": mu_imstar_rup,
            "sigma_lnIMi_lnIMj_rup": sigma_imstar_rup,
            "cov_lnIMi_lnIMj_rup": cov_imstar_rup,
            "weights_imi": weights_imi,
        }
