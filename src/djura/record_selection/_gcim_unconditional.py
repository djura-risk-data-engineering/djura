# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from typing import List

from ._gcim import _GCIM
from ._gcim_select import _GCIMSelect


class _GCIMUnconditional(_GCIM, _GCIMSelect):

    def __init__(self, metadata) -> None:

        self.realization = None
        self.selected_scaled_total: dict = {}
        self.selected_scaled_best: dict = {}
        self.output_create: dict = {
            "target": None,
            "num_components": None,
            "component_definition": None,
        }

        _GCIM.__init__(self, metadata)
        _GCIMSelect.__init__(self, self.output_create, self.metadata)

    def create(
        self,
        gmms: dict,
        ruptures: dict,
        total_weights: dict,
        imi: List[str],
        num_components: int,
        component_definition: str,
    ):
        num_components = int(num_components)

        """Create unconditional GCIM distribution

        Parameters
        ----------
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
        self.output_create["num_components"] = num_components
        if num_components == 1:
            component_definition = "RotD50"

        self.output_create["component_definition"] = component_definition

        # Initialize all cases and periods associated with IMs of interest
        # _{IMi|rup} - IMi given rupture - unconditional spectrum for IMi
        scenarios, imi, mu_rup, sigma_rup, cov_rup, gmms_dict = \
            self._initialize_create(ruptures, imi, gmms, total_weights)

        for im, _scenarios in scenarios.items():
            for i, scenario in enumerate(_scenarios):
                scenario["im_name"] = im

                # Estimate means and stddevs
                # \mu_{ln(IMi|rup)}, \sigma_{ln(IMi|rup)}
                mu_rup[im][:, i], sigma_rup[im][:, i] = \
                    self._get_all_means_stds(
                        imi[im], scenario, component_definition,
                        num_components)

        # Logarithmic mean of target spectrum accounting for
        # all cases (GMMs and rupture scenarios) through contribution factors
        # Exact approach
        # Initialize target mean distribution
        # \mu_{lnIMi}
        mu_exact = {}
        weights_imi = {}
        for im_type, means in mu_rup.items():
            weights = self._get_par_from_scenarios("w", scenarios[im_type])
            weights_imi[im_type] = weights
            mu_exact[im_type] = self.calc_exact_mean(means, weights)

        # Compute covariance matrix for each case
        corr_dict, corr_arr, _im_idxs = self._get_imi_correlation_matrix(imi)

        # COV_{ln(IMi|rup)}
        cov_rup = self._get_all_imi_cov_matrices(
            corr_dict, sigma_rup, cov_rup
        )

        sigma_exact = {}
        cov_exact = {}
        for im_type, means in mu_rup.items():
            sigma_exact[im_type], cov_exact[im_type] = \
                self.calc_exact_sigma_cov(
                means, cov_rup[im_type], mu_exact[im_type],
                weights_imi[im_type]
            )

        # Collect the output
        target = {
            "mu_lnIMi": mu_exact,
            "sigma_lnIMi": sigma_exact,
            "cov_lnIMi": cov_exact,
            "IMi": imi,
            "correlations": corr_dict,
            "combined_correlations": corr_arr,
            "im_idxs": _im_idxs,
        }
        self.output_create["target"] = target

        # Intermediate results
        self.output_create["data"] = {
            "mu_lnIMi_rup": mu_rup,
            "sigma_lnIMi_rup": sigma_rup,
            "cov_lnIMi_rup": cov_rup,
            "weights_imi": weights_imi,
        }
