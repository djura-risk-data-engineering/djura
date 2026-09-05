# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from typing import Tuple, List
from itertools import combinations
from scipy.interpolate import interp1d
import numpy as np

from ..utilities import get_func_args

from .utilities import get_period_im, get_list_id, select_func_args, \
    select_function
from .constants import SUPPORTED_IMS, CORRELATION_MODELS, \
    is_correlation_supported, get_compatible_ims
from .gmm_tools import calculate_epsilon
from .gsim.oq import OQ
from .flatfile import Flatfile
from . import correlation_models
from .gsim import const
from .gsim import imt


class _GCIM:
    NEGLIGIBLE = 1e-16

    def __init__(
        self, metadata: dict, use_disaggregation: bool = False
    ) -> None:
        self.metadata = metadata
        self.USE_FULL_PSHA = use_disaggregation

        #: Correlation model applied to each IM pair, populated during the
        #: creation of the distribution and reported with its output
        self.correlation_models_used = {}

    def _add_missing_im(
            self, im: str, period: float) -> None:
        """Add missing intensity measure to the metadata

        Parameters
        ----------
        im : str
            Intensity measure name
        period : float
            Period of interest
        """
        if period is None or period <= 0.0:
            return

        flatfile = Flatfile(self.metadata)
        flatfile.add_missing_im(im, period)
        self.metadata = flatfile.metadata

    def _validate_gmm(self, gmm: str, **kwargs):
        return OQ()._validate_gmm(gmm, **kwargs)

    def _validate_gmm_indirect_sa_avg(self, gmm: str, **kwargs):
        return OQ()._validate_gmm_indirect_sa_avg(gmm, **kwargs)

    def _gmpe_sb_2014_ratios(self, periods: np.ndarray) -> Tuple[float, float]:
        """Computes Sa_RotD100/Sa_RotD50 ratios.

        References
        ----------
        Shahi, S. K., and Baker, J. W. (2014). "NGA-West2 models for ground-
        motion directionality." Earthquake Spectra, 30(3), 1285-1300.

        Parameters
        ----------
        periods : np.ndarray
            Period(s) of interest (sec)

        Returns
        -------
        Tuple containing
            ratio : float
                geometric mean of Sa_RotD100/Sa_RotD50
            sigma : float
                standard deviation of log(Sa_RotD100/Sa_RotD50)
        """
        # Model coefficient values from Table 1 of the above-reference paper
        periods_orig = np.array(
            [0.01, 0.02, 0.03, 0.05, 0.075,
             0.10, 0.15, 0.20, 0.25, 0.30,
             0.40, 0.50, 0.75, 1.0, 1.5,
             2, 3, 4, 5, 7.50, 10.0])
        mu_ratios_orig = np.array(
            [1.192438059, 1.191246217, 1.187677833, 1.186490749, 1.187677833,
             1.187677833, 1.199614194, 1.205627285, 1.216526905, 1.218962394,
             1.228753204, 1.228753204, 1.237384651, 1.241102379, 1.242344102,
             1.243587068, 1.247323431, 1.259859239, 1.264908769, 1.285310084,
             1.294338819])
        sigma_orig = np.array(
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
             0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

        # Interpolate to compute values for the user-specified periods
        mu_ratio = interp1d(
            np.log(periods_orig), mu_ratios_orig)(np.log(periods))
        sigma = interp1d(
            np.log(periods_orig), sigma_orig)(np.log(periods))

        return mu_ratio, sigma

    def _set_contexts(
        self, case: dict
    ):
        """Sets the parameters for the computation of a ground motion model. If
        not defined by the user as input parameters, most parameters (dip,
        hypocentral depth, fault width, ztor, azimuth, source-to-site distances
        based on extended sources, z2pt5, z1pt0) are defined according to the
        relationships included in Kaklamanos et al. 2011.

        References
        ----------
        Kaklamanos J, Baise LG, Boore DM. (2011) Estimating unknown
        input parameters when implementing the NGA ground-motion prediction
        equations in engineering practice. Earthquake Spectra 27: 1219-1235.
        https://doi.org/10.1193/1.3650372.

        Parameters
        ----------
        case : dict
            Rupture scenario data

        Returns
        -------
        sctx : contexts.SitesContext
            An instance of SitesContext with sites information to calculate
            PoEs on.
        rctx : contexts.RuptureContext
            An instance of RuptureContext with a single rupture information.
        dctx : contexts.DistancesContext
            An instance of DistancesContext with information about the
            distances between sites and a rupture.
        """

        return OQ()._set_contexts(case)

    def get_conditional_im(
        self,
        im_star: dict,
        rupture: dict,
        gmms: dict,
        component_definition: str,
        num_components: int,
    ):
        """Estimates means and standard deviations for IM_star (conditional IM)

        Parameters
        ----------
        im_star : dict
            Conditional IM
        rupture : dict
            Rupture scenario
        gmms : dict
            Ground motion models
        component_definition : str
            Componetn definition
        num_components : int
            Number of componetns, 1 or 2

        Returns
        -------
        dict
            Means
        dict
            Standard deviations
        dict
            Weights of GMMs of IM* associated with the rupture scenario

        Raises
        -------
        ValueError
            If IM* was not defined for IMi and for the specific GMM ID
        """
        im_star_type = im_star["type"]

        gmm_ids = rupture.get("gmms")
        gmm = {}
        for gmm_id in gmm_ids:
            gmm = get_list_id(gmms, "ID", gmm_id, "GMM")
            if im_star_type in gmm:
                break

        if not gmm or im_star_type not in gmm:
            raise ValueError(
                f"IM* type: {im_star_type}, was not defined for IMi GMM"
                f" {gmm_id}"
            )

        if "period" in im_star and im_star['period'] is not None:
            period = [im_star["period"]]
        else:
            period = []

        weights = {}
        means = {}
        sigmas = {}
        for h, _bgmpe in enumerate(gmm[im_star_type]["bgmpe"]):
            gmm_name = gmm[im_star_type]["names"][h]
            weights[gmm_name] = gmm[im_star_type]["weights"][h]

            # GMM name, necessary for estimation of several parameters
            rupture["gmm"] = gmm_name
            rupture["bgmpe"] = _bgmpe
            rupture["im_name"] = im_star["type"]

            _mean, _sigma = self._get_all_means_stds(
                period, rupture, component_definition,
                num_components
            )

            means[gmm_name] = _mean
            sigmas[gmm_name] = float(np.ravel(_sigma)[0])

        return means, sigmas, weights

    def _initialize_create(
        self, ruptures: List[dict], imi: List[str], gmms: List[dict],
        total_weights: dict,
    ):
        """Prepare arguments for filling with outputs and
        mid-calculation results

        Parameters
        ----------
        ruptures : List[dict]
            Rupture scenarios, associated context parameters and weights
        imi : List[str]
            IMis to be considered
        gmms : List[dict]
            GMMs and associated weights

        Returns
        -------
        dict
            Rupture scenarios with corresponding GMMs
        dict
            IMi and associated periods
        dict
            Means, IM: ((len(periods), len(ruptures)))
        dict
            Standard deviations, IM: ((len(periods), len(ruptures)))
        dict
            Covariance matrices,
            IM: ((len(ruptures), len(periods), len(periods)))

        Raises
        ------
        ValueError
            If IM is not supported
        """

        # Retrieve periods of interest associated with corresponding IMs
        im_processed = {}
        for im in imi:
            im_name, im_period = get_period_im(im)

            if im_period is None:
                # Period-independent IM
                im_processed[im_name] = []
                continue

            if im_name not in im_processed:
                im_processed[im_name] = [im_period]
            else:
                im_processed[im_name].append(im_period)

            if im_name not in SUPPORTED_IMS:
                raise ValueError(
                    f"Intensity measure (IM) {im} is not supported. Supported "
                    f"IMs include: {SUPPORTED_IMS}"
                )

        # All scenarios
        # For each scenario gather context parameters, total weight ('w') and
        # GMM ('bmgpe')
        gmms_dict = {}
        for _gmm in gmms:
            gmms_dict = gmms_dict | _gmm

        scenarios = {}
        total_weights_list = {}
        for im in im_processed.keys():
            scenarios[im] = []

            # Transform total weights to list
            if self.USE_FULL_PSHA:
                # This won't be used
                total_weights_list[im] = [0] * \
                    len(ruptures) * len(gmms_dict[im]['names'])
            else:
                total_weights_list[im] = list(total_weights[im])

        for rup_id, _rup in enumerate(ruptures):
            gmm_id = _rup.get("gmms")
            for gmm_id_i in gmm_id:
                gmm = get_list_id(gmms, "ID", gmm_id_i, "GMM")

                for im, data in gmm.items():
                    if im == "ID" or im not in im_processed.keys():
                        continue

                    for name, bgmpe in zip(data["names"], data["bgmpe"]):
                        rup = _rup.copy()
                        rup["gmm"] = name
                        rup["bgmpe"] = bgmpe
                        rup["rup_id"] = rup_id
                        # Only for logic-tree case
                        rup["w"] = total_weights_list[im].pop(0)

                        scenarios[im].append(rup)

        # Conditional mean spectra (in logartihm) for all rupture scenarios
        means = {}
        sigmas = {}
        covariance = {}
        for im, periods in im_processed.items():
            n_scenarios = len(scenarios[im])
            n_periods = len(periods)
            if len(periods) == 0:
                n_periods = 1

            means[im] = np.zeros((n_periods, n_scenarios))
            sigmas[im] = np.zeros((n_periods, n_scenarios))

            # Covariance matrices
            covariance[im] = np.zeros((
                n_scenarios,
                n_periods,
                n_periods,
            ))

        return scenarios, im_processed, means, sigmas, covariance, gmms_dict

    def _get_all_means_stds(
        self, periods, scenario, component_definition, num_components
    ):
        """Calculate and return mean value of intensity distribution and it's
        standard deviation.

        Parameters
        ----------
        periods : List[float]
            Periods associated with period-dependent IMi
        bgmpe : object
            GMPE model
        component_definition : str
            Componetn definition
        num_components : int
            Number of componetns, 1 or 2

        Returns
        -------
        Union[float, List[float]]
            Means
        Union[float, List[float]]
            Standard deviations
        """

        context = self._set_contexts(scenario)

        scenario["ctx"] = context
        scenario["stddev_types"] = [const.StdDev.TOTAL]
        scenario["imt"] = getattr(imt, scenario["im_name"])
        scenario["component_definition"] = component_definition
        bgmpe = scenario["bgmpe"]

        _component = bgmpe.DEFINED_FOR_INTENSITY_MEASURE_COMPONENT
        if isinstance(_component, set):
            gmm_component_name = [c.name.lower() for c in _component]
            rotd100_exists = 'rotd100' in gmm_component_name
        else:
            gmm_component_name = _component.name
            rotd100_exists = 'RotD100' in gmm_component_name

        args = select_func_args(bgmpe.get_mean_and_stddevs, scenario)

        if len(periods) == 0:
            if "imt" in args:
                args["imt"] = args["imt"]()

            # Period-independent IMi
            params = bgmpe.get_mean_and_stddevs(**args)

            try:
                mean = params[0][0]
                sigma = params[1][0][0]
            except IndexError:
                mean = params[0]
                sigma = params[1]

            return mean, sigma

        # Period-dependent IMi
        means = np.zeros(len(periods))
        sigmas = np.zeros(len(periods))

        _imt_def = args["imt"]
        for i, period_i in enumerate(periods):

            # Compute the median prediction through a suitable GMM
            # for the rupture scenario
            # sites, rup, dists, imt, stddev_types
            args["imt"] = _imt_def(period_i)

            params = bgmpe.get_mean_and_stddevs(**args)

            try:
                means[i] = params[0][0]
                sigmas[i] = params[1][0][0].item()
            except IndexError:
                _shape = params[0].shape
                _mean = params[0] if _shape == () else params[0][0]

                means[i] = _mean
                sigmas[i] = params[1]

            # modify spectral targets if RotD100 values were specified for
            # two-component selection
            if component_definition == 'RotD100' and not rotd100_exists \
                    and (num_components == 2 or num_components == 3):
                rotd100_mu_ratio, rotd100_sigma = \
                    self._gmpe_sb_2014_ratios(period_i)

                means[i], sigmas[i] = self._calc_rotd100_mean_sigma(
                    means[i], sigmas[i], rotd100_mu_ratio, rotd100_sigma
                )

        return means, sigmas

    def _calc_rotd100_mean_sigma(self, mean, sigma, mean_ratio, rotd100_sigma):
        mean = mean + np.log(mean_ratio)
        sigma = (sigma ** 2 + rotd100_sigma ** 2) ** 0.5
        return mean, sigma

    def _validate_correlation_models(self, correlation_models: dict = None):
        """Ensure the requested correlation models exist for their IM pairs

        Parameters
        ----------
        correlation_models : dict, optional
            Correlation model to use for each IM pair, by default None

        Raises
        ------
        ValueError
            If a pair is not registered, or the model is not available for it
        """
        if not correlation_models:
            return

        for im_pair_name, model_name in correlation_models.items():
            im_types = im_pair_name.split("-")
            if len(im_types) != 2:
                raise ValueError(
                    f"Correlation model pair {im_pair_name} is not of the "
                    "form 'IMi-IMj'")

            im_i, im_j = im_types
            registered = {im_pair_name, f"{im_j}-{im_i}"} \
                & set(CORRELATION_MODELS)
            if not registered:
                raise ValueError(
                    f"{im_pair_name} is not a supported intensity measure "
                    f"pair. Supported pairs are: {sorted(CORRELATION_MODELS)}")

            available = CORRELATION_MODELS[registered.pop()]
            if model_name not in available:
                raise ValueError(
                    f"Correlation model {model_name} is not available for "
                    f"{im_pair_name}. Available models are: {available}")

    def _validate_correlation_pairs(self, imi: dict, im_star: dict = None):
        """Ensure a correlation model exists for every pair of IMs considered

        The GCIM distribution requires the correlation between each pair of
        intensity measures, so the set of IMs which may be analysed together
        is restricted to those which are mutually supported. An IM lacking a
        correlation equation with another (for example, IA with Sa_avg or
        FIV3) cannot be selected alongside it.

        Parameters
        ----------
        imi : dict
            IMis and associated periods
        im_star : dict, optional
            Conditional IM, by default None (unconditional selection)

        Raises
        ------
        ValueError
            If any pair of the requested IMs has no correlation model
        """
        im_types = list(imi)
        if im_star is not None and im_star["type"] not in im_types:
            im_types.append(im_star["type"])

        missing = sorted({
            f"{min(im_i, im_j)}-{max(im_i, im_j)}"
            for im_i, im_j in combinations(im_types, 2)
            if not is_correlation_supported(im_i, im_j)
        })

        if not missing:
            return

        culprits = sorted({im for pair in missing for im in pair.split("-")})
        details = "\n".join(
            f"  {im} may only be combined with: "
            f"{sorted(get_compatible_ims(im) & set(im_types))}"
            for im in culprits
        )

        raise ValueError(
            "No correlation model is available for the intensity measure "
            f"pair(s) {missing}, so they cannot be analysed together. "
            "Remove one intensity measure of each pair from 'imi', or "
            "choose a different conditioning intensity measure.\n"
            f"{details}"
        )

    def _identify_imi_indexes(self, imi: dict):
        current = 0
        im_idxs = {}
        for im, periods in imi.items():
            n_periods = len(periods) if len(periods) > 0 else 1

            im_idxs[im] = np.arange(current, current + n_periods)
            current += n_periods
        return im_idxs, current

    def _select_correlation_model(
            self, im_pair_name: str, correlation_models: dict = None) -> str:
        """Correlation model to use for a pair of intensity measures

        The model registered first for the pair is used unless the pair is
        named in 'correlation_models', which allows a published study to be
        reproduced with the correlation equations it adopted. The chosen model
        is recorded so that the output states which equations produced it.

        Parameters
        ----------
        im_pair_name : str
            IMi-IMj pair, named as in the correlation model registry
        correlation_models : dict, optional
            Correlation model to use for each IM pair, by default None (the
            first registered model of every pair)

        Returns
        -------
        str
            Name of the correlation model
        """
        model_name = CORRELATION_MODELS[im_pair_name][0]

        if correlation_models:
            im_i, im_j = im_pair_name.split("-")
            for key in (im_pair_name, f"{im_j}-{im_i}"):
                if key in correlation_models:
                    model_name = correlation_models[key]
                    break

        self.correlation_models_used[im_pair_name] = model_name

        return model_name

    def _get_im_star_imi_correlations(
            self, im_star: dict, imi: dict,
            correlation_models: dict = None):
        """Calculates IM* and IMi correlations

        Parameters
        ----------
        im_star : dict
            Conditional IM
        imi : dict
            IMis and associated periods
        correlation_models : dict, optional
            Correlation model to use for each IM pair, by default None

        Returns
        -------
        dict
            Correlation matrices for all IM* and IMi type combinations
        """
        corr = dict()
        im_star_type = im_star["type"]
        if "period" in im_star and im_star['period'] is not None:
            im_star_period = [im_star["period"]]
        else:
            im_star_period = []

        for im, periods in imi.items():

            if im == im_star_type and len(periods) == 0:
                corr[im] = np.asarray([[1.0]])
                continue

            length = len(periods) if len(periods) > 0 else 1
            corr[im] = np.zeros((length, 1))

            model_name = [f"{im}-{im_star_type}", f"{im_star_type}-{im}"]
            models = set(model_name) & set(CORRELATION_MODELS.keys())
            if not bool(models):
                # Guarded by _validate_correlation_pairs, which reports every
                # unsupported pair at once
                raise ValueError(
                    "No correlation model is available for "
                    f"{set(model_name)}, so {im} cannot be conditioned on "
                    f"{im_star_type}")

            # Naming convention in constants
            im_pair_name = models.pop()

            model_name = self._select_correlation_model(
                im_pair_name, correlation_models)

            # Naming convention for proper reading of periods
            im_pair = f"{im}-{im_star_type}"

            corr[im] = self._loop_corr_for_periods(
                periods, im_star_period, model_name, corr[im], im_pair)

            if not isinstance(corr[im], np.ndarray):
                corr[im] = np.asarray([[corr[im]]])

        return corr

    def _get_imi_correlation_matrix(
            self, imi: dict, correlation_models: dict = None) -> np.ndarray:
        """Get correlation matrix for all IMi (unconditioned)

        Parameters
        ----------
        imi : dict
            IMis and associated periods
        correlation_models : dict, optional
            Correlation model to use for each IM pair, by default None

        Returns
        -------
        dict
            Correlation matrix for all IMi categorized
        np.ndarray
            Correlation matrix for all IMi
        dict
            Indices of IMi for populating the matrices
        """
        im_idxs, length = self._identify_imi_indexes(imi)

        _corr = np.zeros((length, length))

        # Populate the correlation matrix
        corr = dict()
        visited = set()
        for im_i, periods_i in imi.items():
            for im_j, periods_j in imi.items():
                _pos = tuple([im_idxs[im_i][:, None], im_idxs[im_j]])

                if im_i == im_j and len(periods_i) == 0 \
                        and len(periods_j) == 0:
                    _corr[_pos] = 1 - self.NEGLIGIBLE
                    continue

                model_name = [f"{im_i}-{im_j}", f"{im_j}-{im_i}"]

                # Check if correlation model is supported
                # if not supported zero correlation is assumed
                models = set(model_name) & set(CORRELATION_MODELS.keys())
                if not bool(models):
                    # Guarded by _validate_correlation_pairs, which reports
                    # every unsupported pair at once
                    raise ValueError(
                        "No correlation model is available for "
                        f"{set(model_name)}, so {im_i} and {im_j} cannot be "
                        "selected together")

                # Naming convention in constants
                im_pair_name = models.pop()
                if im_pair_name in visited:
                    continue

                visited.add(im_pair_name)
                model_name = self._select_correlation_model(
                    im_pair_name, correlation_models)

                # Naming convention for proper reading of periods
                im_pair = f"{im_i}-{im_j}"

                _corr[_pos] = self._loop_corr_for_periods(
                    periods_i, periods_j, model_name, _corr[_pos], im_pair)

                corr[im_pair_name] = _corr[_pos]

        # Ensure symmetry
        # (to avoid double calculation for IMi vs IMj and IMj vs IMi)
        _corr = np.triu(_corr) + np.triu(_corr, 1).T

        return corr, _corr, im_idxs

    def _loop_corr_for_periods(
            self, periods_i, periods_j, model_name, corr, im_pair):
        """Loops for all periods and calculates correlation values

        Parameters
        ----------
        periods_i : List[float]
            Periods of IMi, can be empty for period-independent IMi
        periods_j : List[float]
            Periods of IMj, can be empty for period-independent IMj
        model_name : str
            Correlation model name
        corr : np.ndarray
            Zeros correlation matrix subselection for IMi and IMj pair
        im_pair : str
            IMi-IMj pair

        Returns
        -------
        np.ndarray
            Filled orrelation matrix subselection for IMi and IMj pair
        """
        model = select_function(correlation_models, model_name)

        # ANN model
        ann_model = "im_pair" in get_func_args(model)

        if len(periods_i) == 0 and len(periods_j) == 0:
            # Both IMs are period-independent
            if ann_model:
                return model(im_pair)
            return model()

        if len(periods_i) == 0 or len(periods_j) == 0:
            # Only one IM is period-independent
            periods = periods_i or periods_j

            for i, period in enumerate(periods):
                if ann_model:
                    _corr = model(im_pair, period)
                else:
                    _corr = model(period)

                if len(periods_i) == 0:
                    corr[0][i] = _corr
                else:
                    corr[i] = _corr

            return corr

        # Both IMs are period-dependent
        for i, t1 in enumerate(periods_i):
            for j, t2 in enumerate(periods_j):
                if ann_model:
                    corr[i, j] = model(im_pair, t1, t2)
                else:
                    corr[i, j] = model(t1, t2)

        return corr

    def _get_all_imi_cov_matrices(
        self, corr: dict, sigmas: dict, cov: dict,
    ) -> dict:
        """Get covariance matrix

        Parameters
        ----------
        corr : dict
            Correlation matrices categorized by IMi-IMj type
        sigmas : dict
            Standard deviations
        cov : dict
            Initialized empty covariance matrix

        Returns
        -------
        Filled covariance matrix
            dict, IMi vs IMj where IMi = IMj
        """

        for im in cov.keys():
            im_pair = f"{im}-{im}"

            if im_pair not in corr:
                # Period-independent
                cov[im].fill(1)
                continue

            _corr = corr[im_pair]
            for i in range(_corr.shape[0]):
                for j in range(_corr.shape[1]):
                    cov[im][:, i, j] = _corr[i, j] * \
                        sigmas[im][i] * sigmas[im][j]

            cov[im][cov[im] == 0] += self.NEGLIGIBLE

        return cov

    def _get_conditional_correlation(self, corr, corr_imi_imstar, im_idx):
        """Get conditional correlations
        Pearson product-moment correlation coefficient

        Parameters
        ----------
        corr : np.ndarray
            IMi correlation matrix
        corr_imi_imj : dict
            Conditional correlation for each IMi
        im_idx : dict
            IMi and corresponding indexes for the matrix population

        Returns
        -------
        dict
            Conditional correlation matrixes for each IMi
        np.ndarray
            Conditional correlation matrix
        """
        n_imi = corr.shape[0]

        # Correlations combined for IMi vs IMj
        corr_arr = np.zeros((n_imi, 1))
        for _im_i, _corr in corr_imi_imstar.items():
            im_i_idx = im_idx[_im_i]

            corr_arr[im_i_idx] = np.asarray(_corr)

        # transpose into an array of shape (n_imi, )
        corr_arr = corr_arr.reshape(-1)

        correlations = np.zeros(corr.shape)
        for i in range(n_imi):
            for k in range(n_imi):
                if corr_arr[i] == 1:
                    corr_arr[i] -= self.NEGLIGIBLE
                if corr_arr[k] == 1:
                    corr_arr[k] -= self.NEGLIGIBLE
                if corr[i][k] == 1:
                    corr[i][k] -= self.NEGLIGIBLE

                correlations[i, k] = (
                    corr[i][k] - corr_arr[i] * corr_arr[k]) / \
                    (np.sqrt(1 - corr_arr[i]**2) * np.sqrt(1 - corr_arr[k]**2))

        # Create the dictionary pairs too
        corr_dict = {}
        for imi, indxs in im_idx.items():
            corr_dict[f"{imi}-{imi}"] = correlations[indxs[:, None], indxs]

        return corr_dict, correlations

    def calc_exact_mean(self, means: np.ndarray, weights: List[float]):
        """Computes exact mean given weights

        Parameters
        ----------
        means : np.ndarray
            Means
        weights : List[float]
            Weights for each rupture scenario

        Returns
        -------
        np.ndarray
            Computed exact means
        """
        weights = np.asarray(weights)

        return np.sum(means * weights, 1)

    def _get_exact_sigma(self, sigmas, means, mean_exact, weights):
        """Calculates exact stdevs

        Parameters
        ----------
        sigmas : np.ndarray
            Stdevs
        covs : np.ndarray
            Covariance matrix
        mean_exact : np.ndarray
            Exact mean
        weights : List[float]
            Weights for each rupture scenario

        Returns
        -------
        np.ndarray
            Exact sigmas
        """
        var_imi_case = np.zeros(sigmas.shape)
        for i, weight in enumerate(weights):
            variance = np.square(sigmas[:, i])
            mean = means[:, i]
            var_imi_case[:, i] = weight * (variance + (mean_exact - mean) ** 2)
        sigma_exact = np.sqrt(np.sum(var_imi_case, 1))
        return sigma_exact

    def calc_exact_sigma_cov(
        self, means: np.ndarray, covs: np.ndarray,
        mean_exact: np.ndarray, weights: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes exact sigma and covariance given weights

        Parameters
        ----------
        means : np.ndarray
            Means
        covs : np.ndarray
            Covariance matrix
        mean_exact : np.ndarray
            Exact mean
        weights : List[float]
            Weights for each rupture scenario

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Exact sigmas
            Exact covariance
        """
        # Loop over each case
        var_imi_case = np.zeros((covs.shape[1], covs.shape[0]))

        for i, weight in enumerate(weights):
            variance = np.diagonal(covs[i])
            mean = means[:, i]
            var_imi_case[:, i] = weight * (variance + (mean_exact - mean) ** 2)

        # Case index does not matter, as the diagonal terms will be of interest
        cov_exact = covs[0, :, :]
        var_imi_rup = np.sum(var_imi_case, 1)
        cov_exact[np.eye(cov_exact.shape[0]) == 1] = var_imi_rup

        # Avoid positive semi-definite covariance matrix with several
        # eigenvalues being exactly zero
        # See: https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning!  # noqa
        min_eig = np.min(np.real(np.linalg.eigvals(cov_exact)))
        if min_eig < 0:
            cov_exact -= 2 * min_eig * np.eye(*cov_exact.shape)
            # Add a small multiple of the identity matrix
            # cov_exact += self.NEGLIGIBLE * np.eye(*cov_exact.shape)
        # Get the diagonal, i.e., the variance,
        # then estimate the standard deviation
        sigma_exact = np.sqrt(np.diagonal(cov_exact))

        return sigma_exact, cov_exact

    def _get_supported_parameters(self, which: str):
        return OQ()._get_supported_parameters(which)

    def _get_par_from_scenarios(self, par: str, scenarios: List):
        return [sc[par] for sc in scenarios]

    @staticmethod
    def _get_weight_disaggregation(eps_im_star: dict, gmms: dict):
        ws_sample = np.asarray(next(iter(eps_im_star['ws'].values())))
        if ws_sample.ndim == 1:
            ws_sample = ws_sample[:, np.newaxis]
        weights = np.zeros((len(gmms['names']) * (len(eps_im_star) - 1),
                            ws_sample.shape[1]))

        start_idx = 0
        # The logic tree leaf weight is carried by the values themselves,
        # having been applied where eps_im_star['ws'] was assembled
        for gmm in gmms['names']:
            ws_list = np.asarray(eps_im_star['ws'][gmm])
            if ws_list.ndim == 1:
                ws_list = ws_list[:, np.newaxis]
            end_idx = start_idx + ws_list.shape[0]
            weights[start_idx:end_idx, :] = ws_list
            start_idx = end_idx

        return weights.squeeze()

    def _calculate_epsilon_for_rup(
            self, im_star_value, means, sigmas, weights):

        means = np.array(list(means.values())).T
        sigmas = np.array(list(sigmas.values()))
        weights = np.array(list(weights.values()), dtype=float)

        mean = np.sum(means * weights)

        sigma = np.sqrt(np.sum(
            weights * (sigmas ** 2 + (means - mean) ** 2)))

        return calculate_epsilon(im_star_value, mean, sigma)

    def get_available_gsims(self):
        return OQ().get_available_gsims()

    def check_gmpe_attributes(self, gmpe: str):
        OQ().check_gmpe_attributes(gmpe)

    def get_gmpe_attributes(self, gmpe: str):
        return OQ().get_gmpe_attributes(gmpe)

    @staticmethod
    def _truncnorm_sf(phi_b, epsilon):
        """
        Fast survival function for truncated normal distribution.
        Assumes zero mean, standard deviation equal to one and symmetric
        truncation. It is faster than using scipy.stats.truncnorm.sf.

        phi_b:
            ndtr(truncation_level); assume phi_b > .5
        values:
            Numpy array of values as input to a survival function for the given
            distribution.

        Returns
        ----------
            Numpy array of survival function results in a
            range between 0 and 1.
            For phi_b close to .5 returns a step function 1 1 1 1 .5 0 0 0 0 0.
        """
        from scipy.special import ndtr

        # notation from
        # http://en.wikipedia.org/wiki/Truncated_normal_distribution.
        # given that mu = 0 and sigma = 1, we have alpha = a and beta = b.
        # "CDF" in comments refers to cumulative distribution function
        # of non-truncated distribution with that mu and sigma values.
        # assume symmetric truncation, that is ``a = - truncation_level``
        # and ``b = + truncation_level``.
        # calculate Z as ``Z = CDF(b) - CDF(a)``, here we assume that
        # ``CDF(a) == CDF(- truncation_level) == 1 - CDF(b)``
        z = phi_b * 2. - 1.

        # calculate the result of survival function of ``values``,
        # and restrict it to the interval where probability is defined --
        # 0..1. here we use some transformations of the original formula
        # that is ``SF(x) = 1 - (CDF(x) - CDF(a)) / Z`` in order to minimize
        # number of arithmetic operations and function calls:
        # ``SF(x) = (Z - CDF(x) + CDF(a)) / Z``,
        # ``SF(x) = (CDF(b) - CDF(a) - CDF(x) + CDF(a)) / Z``,
        # ``SF(x) = (CDF(b) - CDF(x)) / Z``.
        return ((phi_b - ndtr(epsilon)) / z).clip(0., 1.)

    @staticmethod
    def _find_gmm_by_key(gmms, im_type, gmm_ids):
        """Entry of the first associated GMM which defines an IM type

        Parameters
        ----------
        gmms : List[dict]
            GMMs and associated weights, each carrying its own 'ID'
        im_type : str
            Intensity measure type to look for
        gmm_ids : List[int]
            IDs of the GMMs associated with a rupture scenario

        Returns
        -------
        dict
            Names and weights registered for 'im_type'

        Raises
        ------
        ValueError
            If none of the associated GMMs defines the intensity measure
        """
        # Resolved through the 'ID' of each entry, as in get_conditional_im,
        # so that the two agree for a 'gmms' list whose IDs are given by the
        # input rather than taken from the position in the list
        for gmm_id in gmm_ids:
            gmm = get_list_id(gmms, "ID", gmm_id, "GMM")
            if im_type in gmm:
                return gmm[im_type]

        raise ValueError(
            f"IM* type: {im_type}, was not defined for any of the GMMs "
            f"associated with the rupture scenario: {list(gmm_ids)}")

    @staticmethod
    def _identify_indirect_sa_avg(avg_sa, im_star, imi, gmms, total_weights):
        # Will use only one type of Sa_avg
        # So, if the list includes different t_high for example, it will
        # still default to the first index and ignore any other inputs
        t_high = None

        for i, gmm in enumerate(gmms):
            if "Sa_avg" in gmm:

                t_high = int(avg_sa['t_high'])
                gmm[f"Sa_avg{t_high}"] = gmm.pop("Sa_avg")
                gmms[i] = gmm

        if t_high is None:
            return im_star, imi, gmms, total_weights

        if total_weights is not None and "Sa_avg" in total_weights:
            total_weights[f"Sa_avg{t_high}"] = total_weights.pop("Sa_avg")

        if im_star is not None and im_star["type"] == "Sa_avg":
            im_star["type"] = f"Sa_avg{t_high}"

        if isinstance(imi, dict) and imi["type"] == "Sa_avg":
            imi["type"] = f"Sa_avg{t_high}"
        else:
            for i, im in enumerate(imi):
                if "Sa_avg" in im:
                    imi[i] = im.replace("Sa_avg", f"Sa_avg{t_high}")

        return im_star, imi, gmms, total_weights
