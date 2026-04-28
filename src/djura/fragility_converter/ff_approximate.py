from pathlib import Path
from typing import List, Union
from scipy.integrate import simpson
import numpy as np

from ..record_selection.gmm_tools import (
    calculate_epsilon, get_mean_xy, get_sigma_xy
)
from ..record_selection.utilities import get_list_id, get_period_im
from .ff import IMModel, FF


class FFApproximate(FF):
    NEGLIGIBLE = 1e-9

    def __init__(
        self,
        im1: IMModel,
        im2: IMModel,
        data: Union[Path, str, dict] = None,
        correlation_type: str = None,
    ) -> None:
        FF.__init__(self, im1, im2, data, None, correlation_type)

    def create(self):
        return self._create(**self._create_input)

    def _validate_create_input(self):
        """Validate input arguments for create method
        """
        if "component-definition" in self.data:
            if self.data["component-definition"] == "rotd50":
                self.data["component-definition"] = "RotD50"
            if self.data["component-definition"] == "rotd100":
                self.data["component-definition"] = "RotD100"

        # Ensure GMM is supported by engine
        for _id, gmms in enumerate(self.data["gmms"]):
            for im, val in gmms.items():
                if im == "ID":
                    continue

                bgmpes = []
                src_gmms = [f"{s}_{ii}" for ii, s in enumerate(val["names"])]
                kwargs = val.get('kwargs', None)
                if kwargs is None:
                    kwargs = [{}] * len(src_gmms)

                for name, kwarg in zip(src_gmms, kwargs):
                    if im == "Sa_avg":
                        avg_sa_kwarg = self.data['avg-sa']
                        # Indirect Sa_avg
                        bgmpes.append(
                            self._validate_gmm_indirect_sa_avg(
                                self._validate_gmm(name, **kwarg),
                                **avg_sa_kwarg
                            ))
                    else:
                        bgmpes.append(
                            self._validate_gmm(name, **kwarg))

                gmms[im]["bgmpe"] = bgmpes
                gmms[im]["names"] = src_gmms

            if "ID" not in gmms:
                gmms["ID"] = _id

        # Target IM (IM2)
        im_type, period = get_period_im(self.im2["name"])
        self.im2['type'] = im_type
        self.im2['period'] = period

        # Rupture distance estimation
        site_parameters = self.data.pop("site-parameters", {})

        for site_key in site_parameters:
            try:
                site_parameters[site_key] = float(site_parameters[site_key])
            except ValueError:
                site_parameters[site_key] = site_parameters[site_key]

        for _val in self.data['poes'].values():
            ruptures = _val.get("ruptures")
            for i, rupture in enumerate(ruptures):
                if "z_tor" in rupture.keys() and "rrup" not in rupture.keys():
                    ruptures[i]['rrup'] = (
                        rupture['rjb'] ** 2 + rupture["z_tor"] ** 2) ** 0.5
                if "rake" not in rupture.keys():
                    ruptures[i]["rake"] = 0.0

                if not self.USE_FULL_PSHA:
                    ruptures[i]["gmms"] = np.arange(
                        len(self.data["gmms"])).tolist()

                ruptures[i] = {
                    **ruptures[i],
                    **site_parameters
                }

        # Approximate approach
        # Get ruptures from corresponding POEs
        # each rupture list weight will be normalised per POE level
        self.im2_poes = []
        self.im2_range = []

        for poe, _val in self.data['poes'].items():
            ruptures = _val.get("ruptures")
            self.im2_poes.append(float(poe))
            self.im2_range.append(_val["im-star"]["value"])

            # Sort using im2 range
            self.data['poes'][poe]["total-weights"] = self._validate_weights(
                ruptures, self.data["gmms"])

        self.im2_range, self.im2_poes = zip(*sorted(
            zip(self.im2_range, self.im2_poes)))

    def _validate_weights(self, ruptures: List[dict], gmms: List[dict]):
        """Validate weights of rupture scenarios

        Parameters
        ----------
        ruptures : List[dict]
            Rupture scenarios
        gmms : List[dict]
            GMMs associated with IM types

        Returns
        -------
        dict
            Total hazard contributions per IM1
        """
        total_weights = self.data.get("total-weights", dict())

        # Number of scenarios
        scenario_counter = {}
        for rup in ruptures:
            for rup_gmm in rup["gmms"]:
                gmm = get_list_id(gmms, "ID", rup_gmm, "GMM")
                for im, gmm_val in gmm.items():
                    if im == "ID":
                        continue

                    if im not in scenario_counter:
                        scenario_counter[im] = len(gmm_val["names"])
                    else:
                        scenario_counter[im] += len(gmm_val["names"])

        # Hazard contributions (total-weights) not provided
        # inferring from logic tree of ruptures and GMMs for each IM1
        total_weights = {}
        for rup in ruptures:
            # Query for associated GMMs
            for rup_gmm in rup["gmms"]:
                gmm = get_list_id(gmms, "ID", rup_gmm, "GMM")

                # Rupture weights
                w_rup = rup.get("weight", 1 / len(ruptures))

                if w_rup == 0:
                    w_rup = 1 / len(ruptures)

                for im, gmm_val in gmm.items():
                    if im == "ID":
                        continue

                    # Check if any weight was provided
                    # if none provided, assume equal weighting for GMMs
                    n_gmms = len(gmm_val["names"])
                    w_gmms = gmm_val.get("weights", [1 / n_gmms] * n_gmms)
                    w_gmms = np.array(w_gmms, dtype=float)
                    w_tot = np.multiply(w_rup, w_gmms)

                    if im in total_weights:
                        total_weights[im] = np.concatenate((total_weights[im],
                                                            w_tot))
                    else:
                        total_weights[im] = w_tot

                    gmm[im]["weights"] = w_gmms

        # Ensure that total weights sum up to 1
        for im, w in total_weights.items():
            total_weights[im] = w / np.sum(w)

        self._ensure_total_weights_length(scenario_counter,
                                          total_weights)

        return total_weights

    def _pop_keys(self) -> dict:
        """Prepare input keys for create and select methods

        Parameters
        ----------
        method : str
            "create" or "select"

        Returns
        -------
        dict
            self.data updated
        """

        create_vars = [
            "gmms", "poes", "num_components",
            "component_definition", "total_weights", "avg_sa",
        ]

        data = {}
        for key, value in self.data.items():
            key = key.replace("-", "_")
            if key in create_vars:
                data[key] = value
        return data

    def _create(
        self,
        gmms: dict,
        poes: dict,
        total_weights: dict,
        num_components: int,
        component_definition: str,
        avg_sa: dict = None,
    ):
        num_components = int(num_components)

        self.output_create["num_components"] = num_components
        if num_components == 1:
            component_definition = "RotD50"

        self.output_create["component_definition"] = component_definition

        if avg_sa is not None:
            # TODO, not implemented yet
            self.im2, self.im1, gmms, total_weights = \
                self._identify_indirect_sa_avg(
                    avg_sa, self.im2, self.im1, gmms, total_weights
                )

        # im2_range = np.asarray(self.im2['value'])
        im2_range = np.asarray(self.im2_range)
        im2_probs = np.zeros(im2_range.shape)

        for poe, _val in poes.items():

            _scenarios, imi, _mu_rup, _sigma_rup, _, _ = \
                self._initialize_create(
                    _val['ruptures'], [self.im1['name']], gmms,
                    _val['total-weights']
                )

            poes[poe]['scenarios'] = _scenarios
            poes[poe]['mu_rup'] = _mu_rup
            poes[poe]['sigma_rup'] = _sigma_rup

            mu_imstar_rup = {}
            sigma_imstar_rup = {}
            for _im, arr in _mu_rup.items():
                mu_imstar_rup[_im] = np.zeros(arr.shape)

                sigma_imstar_rup[_im] = np.zeros(_sigma_rup[_im].shape)

            poes[poe]["mu_imstar_rup"] = mu_imstar_rup
            poes[poe]["sigma_imstar_rup"] = sigma_imstar_rup

        # Compute correlation coefficients for IM2 and IM1s of interest
        # \rho_{lnIM1,lnIM2|Rup}
        _corr_model = self.correlation_type \
            if self.correlation_type == "eshm20" else None
        rho_imi_im_star = self._get_im_star_imi_correlations(
            self.im2, imi, _corr_model)

        # Conditioned on IM2, {rup_i: [gmm1, gmm2, ..., gmmn]}
        mu_im_star = {}
        sigma_im_star = {}
        eps_im_star = {}
        eps_im_star_comb = {}
        weights_im_star = {}
        for poe, _val in poes.items():
            _ruptures = _val["ruptures"]
            mu_im_star[poe] = {}
            sigma_im_star[poe] = {}
            eps_im_star[poe] = {}
            eps_im_star_comb[poe] = {}
            weights_im_star[poe] = {}
            for i, _rupture in enumerate(_ruptures):
                mu_im_star[poe][i], sigma_im_star[poe][i], \
                    weights_im_star[poe][i] = \
                    self.get_conditional_im(
                    self.im2, _rupture, gmms, component_definition,
                    num_components)

        # Back-calculate epsilon at the IM2
        # Shape = (len(ruptures), len(im_star_gmms))
        im_star_gmms = {}
        for j, (poe, _val) in enumerate(mu_im_star.items()):
            # For each POE level (IM2 level)
            im_star_gmms[poe] = set()
            for i, _mu_im_star in _val.items():
                # For each scenario
                im_star_gmms[poe] = im_star_gmms[poe].union(
                    set(_mu_im_star.keys()))
                eps_im_star[poe][i] = {}
                for g, _mu in _mu_im_star.items():
                    eps_im_star[poe][i][g] = calculate_epsilon(
                        im2_range[j], _mu, sigma_im_star[poe][i][g])

        # IM1
        _imt1, _t1 = get_period_im(self.im1['name'])
        self.im1['type'], self.im1['period'] = _imt1, _t1

        im = self.im1['type']

        if self.correlation_type == "high":
            for _im in rho_imi_im_star.keys():
                rho_imi_im_star[_im] = rho_imi_im_star[_im] ** (1 / 2)

        elif self.correlation_type == "one":
            for _im in rho_imi_im_star.keys():
                rho_imi_im_star[_im] = np.array([0.99])

        # IMi calculations (IM1 calculations)
        for poe, _val in poes.items():
            scenarios = _val["scenarios"]

            for i, scenario in enumerate(scenarios[im]):

                # Retrieve epsilon
                epsilon = eps_im_star[poe][scenario["rup_id"]][scenario["gmm"]]

                scenario["im_name"] = im

                # Estimate means and stddevs conditioned
                # on rupture scenario, \mu_{ln(IM1|rup)}, \sigma_{ln(IM1|rup)}
                poes[poe]["mu_rup"][im][:, i], \
                    poes[poe]["sigma_rup"][im][:, i] = \
                    self._get_all_means_stds(
                        imi[im], scenario, component_definition,
                        num_components)

                # Compute \sigma_{ln(IM1|IM2,rup)}
                poes[poe]["sigma_imstar_rup"][im][:, i] = get_sigma_xy(
                    poes[poe]["sigma_rup"][im][:, i], rho_imi_im_star[im])

                # Compute \mu_{ln(IM1|IM2,rup)}
                # Initially compute for each GMM case of IM2
                # shape=(len(gmms_IM2), len(im1[im]))
                poes[poe]["mu_imstar_rup"][im][:, i] = get_mean_xy(
                    poes[poe]["mu_rup"][im][:, i],
                    poes[poe]["sigma_rup"][im][:, i],
                    rho_imi_im_star[im],
                    epsilon
                )

        # Logarithmic median of target spectrum accounting for
        # all cases (GMMs and rupture scenarios) through contribution factors
        # Exact approach
        weights_imi = {}
        for poe, _val in poes.items():
            mu_imstar_rup = _val["mu_imstar_rup"]
            scenarios = _val["scenarios"]
            weights_imi[poe] = {}
            for im_type, _ in mu_imstar_rup.items():
                # Logic tree weights
                weights_imi[poe][im_type] = self._get_par_from_scenarios(
                    "w", scenarios[im_type])

        # Site-specific probability density function is computed using
        # mu_imstar_rup = mu_{lnIM1|lnIM2,rup}, shape=(1, n_rups, im2.num_pts)
        # sigma_imstar_rup = sigma_{lnIM1|lnIM2,rup}, shape=(1, n_rups)
        im_type = self.im1['type']
        # sigmas = None
        sigmas = []
        mus = []
        weights = []
        max_scenario_per_poe = max(len(weights_imi[poe][im_type])
                                   for poe in poes)
        for poe, _val in poes.items():
            sigma_imstar_rup = _val["sigma_imstar_rup"]
            mu_imstar_rup = _val["mu_imstar_rup"]

            mu = np.squeeze(np.exp(mu_imstar_rup[im_type]), axis=0)
            sigma = np.squeeze(sigma_imstar_rup[im_type], axis=0)

            if len(weights_imi[poe][im_type]) < max_scenario_per_poe:
                # Ensure same length for all POE scenarios for matrices
                # any added scenario will receive a weight of zero to
                # avoid impacting the results
                _n_to_add = max_scenario_per_poe - \
                    len(weights_imi[poe][im_type])
                weights_imi[poe][im_type].extend([0] * _n_to_add)
                mu = np.append(mu, np.full(_n_to_add, mu[-1], dtype=mu.dtype))
                sigma = np.append(sigma, np.full(_n_to_add, sigma[-1],
                                                 dtype=sigma.dtype))

            # Combine and normalize all weights
            weights.append(weights_imi[poe][im_type])
            mus.append(mu)
            sigmas.append(sigma)

        weights = np.asarray(weights, dtype=float).T
        sigma_all = np.asarray(sigmas, dtype=float).T
        mu_all = np.asarray(mus, dtype=float).T

        im1_range, im1_probs, pdfs = self._get_im1_cdf(sigma_all, mu_all)

        # Compute pff in matrix form (element-wise multiplication)
        # im1_probs: (K,) -> broadcast to (L, 1, 1)
        # pdfs: (K, M, N)
        # weights_imi[im_type]: (M, N)
        # pff: (K, M, N)
        # K: Length of IM1 values
        # M: Number of scenarios
        # N: Number of IM2 values or POEs
        pff = im1_probs[:, None, None] * pdfs * weights
        # Integrate using Simpson's rule along im1_range axis (axis=0) to
        # reduce to (M, N)
        integrated_pff = simpson(pff, x=im1_range, axis=0)  # (M, N)
        # Sum over M axis to get im2_probs (N,)
        im2_probs = np.sum(integrated_pff, axis=0)  # (N,)

        # -------
        # Ensure probabilities do not decrease because of
        # missing computational data at very high IMs
        im2_probs = np.maximum.accumulate(im2_probs)

        # Limit maximum values to 1
        # a problem can occur for low DS where im range does not capture
        # many data points at lower intensity values
        im2_probs = np.clip(im2_probs, None, 1)

        return im2_probs, im2_range
