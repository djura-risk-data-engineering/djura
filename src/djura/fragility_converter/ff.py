from pathlib import Path
import yaml
import json
from typing import List, Union
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpy as np
from pydantic import BaseModel, Field

from ..record_selection.gmm_tools import (
    calculate_epsilon, get_mean_xy, get_sigma_xy
)
from ..record_selection.utilities import get_period_im
from ..record_selection._gcim import _GCIM


_IMS_ACC = ['PGA', 'SA', 'Sa_avg2', 'Sa_avg3']
_IMS_VEL = ['PGV', 'FIV3']


class IMModel(BaseModel):
    # Required only for known IM, i.e., IM1
    median: float = Field(default=None)
    dispersion: float = Field(default=None)
    im_range: List[float] = Field(default=None)
    probs: List[float] = Field(default=None)

    # e.g. SA(0.1), FIV3(0.1), PGA, Sa_avg2(0.1) etc.
    name: str

    # For IM2 only
    min: float = Field(default=None)
    max: float = Field(default=None)
    num_pts: int = Field(default=None)

    value: List[float] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Generate the values during initialization
        if all(getattr(self, attr) is not None
               for attr in ['min', 'max', 'num_pts']):
            self.value = list(np.linspace(self.min, self.max, self.num_pts))
            # self.value = list(
            #     np.logspace(np.log10(0.001), np.log10(5), num=40))


class FF(_GCIM):
    NEGLIGIBLE = 1e-9

    # Default input arguments
    default_data: dict = {
        # create() arguments
        "gmms": None,
        # [{
        #     "ID": Optional[int],
        #     "IM-type": {
        #         "names": [],
        #         "weights": Optional([]),
        #         "total-weights": Optional([]),
        #     },
        # }, ... next one
        # ]
        "ruptures": None,
        # [
        #     {"mag": float, "weight": Optional(float), "gmms": [ID]},
        #     ... next one
        # ]
        # Total weights sequence: rup1 - gmms, rup2 - gmms, ...
        "total-weights": None,
        "num-components": 2,
        "component-definition": "RotD50",
        "imi": [
            "SA(0.05s)", "SA(0.075s)", "SA(0.1s)", "SA(0.15s)",
            "SA(0.2s)", "SA(0.25s)", "SA(0.3s)", "SA(0.4s)",
            "SA(0.5s)", "SA(0.75s)", "SA(1.0s)", "SA(1.3s)",
            "SA(1.5s)", "SA(2.0s)"],
        # create() arguments Conditional-based GCIM
        "im-star": None,
        # {
        #     "type": None,
        #     "value": None,
        # },
        # select() arguments
        "greedy-loops": 1,
        "nreplicate": 1,
        "num_records": 40,
        "context_limits": {
            # Those keys must be present in metadata file,
            # otherwise the limits will be ignored
            "magnitude": None,
            "soil_Vs30": None,
            "Rjb": None,
            "Rrup": None,
            "mechanism": None,
        },
        # Needed for indirect avg sa computation
        "avg-sa": None,
        "seed": 0,
        "ks_alpha": 0.05,
        "im_weights": [],
        "max_scaling_factor": 3.0,
        "add_data_for_dis": None,
    }

    def __init__(
        self,
        im1: IMModel,
        im2: IMModel,
        data: Union[Path, str, dict] = None,
        dis_oq: dict = None,
        correlation_type: str = None,
    ) -> None:
        im1 = IMModel.model_validate(im1).model_dump()
        im2 = IMModel.model_validate(im2).model_dump()

        self.output_create: dict = {
            "target": None,
            "num_components": None,
            "component_definition": None,
        }

        self.USE_FULL_PSHA = False
        self.correlation_type = correlation_type

        # IM1 and IM2
        self.im1 = im1
        self.im2 = im2

        # Data
        self.data: dict = None

        # Validate input parameters
        if data is not None:
            self._read_input_file(data)

        # OQ Engine disaggregation data
        self.dis_oq = dis_oq
        if dis_oq is not None:
            self._read_from_oq_dis(dis_oq)
            self.USE_FULL_PSHA = True

        _GCIM.__init__(self, None, self.USE_FULL_PSHA)

        self._validate_create_input()
        self._create_input = self._pop_keys()

    def _read_from_oq_dis(self, dis_oq):
        """
        Processed disaggregation data from OQ Engine analysis
        Overrides:
            gmms; site-parameters; ruptures; imi; im-star

        Notice
        ----------
        Currently implemented for backend and development purposes only!

        Keys
        ----------
        im_ref : str
            IM2 name, value will be inferred from hazard curves
        hazard-curves : ndarray
            Hazard curves
        imt : dict
            IMT: ndarray, IMTs and values at each POE used during
            OQ Engine analysis
        oq-poes : Listr[str]
            List of POEs used during OQ Engine analysis
        site-parameters : List[str]
            Required site parameters
        ctx_by_grp : dict
            source model ID : numpy.recarray[numpy.record]
            To access for example 'mag': ctx.mag
        lt-weights : List[float]
            Logic tree leaf weights = gmm_wegith * source_model_weight
        gsims : List[str]
            GSIM names
        gsim-weights : List[float]
            GMM weights, same as lt-weights?
        totrups : int
            Total number of ruptures (includes all sources)
        required-parameters : set(str)
            Required rupture and distance parameters
        phi_b : float
            0.5 * (1.0 + math.erf(z * sqrt(0.5)))
            where z is truncation level
        invtime : float
            Investigation time
        data : dict
            Additional data for each source model ID
            For details see below

        Keys of 'data' per source model ID
        ----------
        gsims : dict
            GMM : ID pairs for source model
        """

        # IMLs
        self.data['imi'] = list(dis_oq['imt'].keys())

        if self.im1['name'] not in self.data['imi']:
            raise ValueError(f"{self.im1['name']} unavailable")

        if self.im2['name'] not in self.data['imi']:
            raise ValueError(f"{self.im2['name']} unavailable")
        self.data['imi'] = [self.im1['name']]

        # Context parameters
        ctx_by_grp = dis_oq['ctx_by_grp']

        # Site parameters
        site_params = {}
        for param in dis_oq['site-parameters']:
            site_params[param] = ctx_by_grp[0][param][0]

        self.data['site-parameters'] = site_params

        # GMMs and logic tree leaf weights
        # lt_weight = gmm_weight * source_model_weight
        lt_weights = dis_oq['lt-weights']
        _imt1, _t1 = get_period_im(self.im1['name'])
        self.im1['type'], self.im1['period'] = _imt1, _t1

        _imt2, _ = get_period_im(self.im2['name'])
        _imts = set([_imt1]) | set([_imt2])

        gmms = []
        self.data['avg-sa'] = None

        for gsims in dis_oq['data'].values():
            src_gmms = []
            weight_ids = []
            gsim_kwargs = []
            for _gsim, _kwargs in zip(gsims['gsims'], gsims['parameters']):
                src_gmms += list(_gsim.keys())
                weight_ids += list(_gsim.values())
                gsim_kwargs.append(_kwargs)

            src_gmm_weights = lt_weights[weight_ids].reshape(1, -1)[0]

            for imt in _imts:
                if imt == "Sa_avg" and self.data['avg-sa'] is None:
                    self.data['avg-sa'] = gsim_kwargs[0]

                # In current implementation, all GMMs must be applicable
                # to all IMTs
                gmms.append({
                    imt: {
                        "names": src_gmms,
                        "weights": src_gmm_weights,
                        'kwargs': gsim_kwargs,
                    }
                })
        self.data['gmms'] = gmms

        # Ruptures (distance and rupture context parameters)
        req_params = dis_oq['required-parameters']
        ruptures = []

        for ctx_id, ctx in enumerate(ctx_by_grp.values()):
            gmm_ids = list(range(
                ctx_id * len(_imts),
                (ctx_id + 1) * len(_imts)
            ))

            for row in ctx:
                ctx_i = {}
                for param in req_params:
                    ctx_i[param] = getattr(row, param)

                # Add probs and occurrence rates too
                ctx_i['occurrence_rate'] = getattr(row, 'occurrence_rate')
                ctx_i['probs'] = getattr(row, 'probs')
                ctx_i['gmms'] = gmm_ids
                ruptures.append(ctx_i)

        self.data['ruptures'] = ruptures
        self.data['add_data_for_dis'] = {
            "inv_time": dis_oq['invtime'],
            "phi_b": dis_oq["phi_b"],
        }

    def _read_input_file(self, filename: Union[Path, dict]):
        """Reads input data file and updates the default inputs
        accordingly

        Parameters
        ----------
        filename : Union[Path, dict]
            Path to datafile or datafile content as a dict
        """
        self.data = self.default_data.copy()
        if isinstance(filename, dict):
            self.data.update(filename)
            return

        if filename.suffix == ".json":
            with open(filename) as f:
                data = json.load(f)
        else:
            # yaml or yml
            with open(filename, "r") as f:
                data = yaml.safe_load(f)

        self.data.update(data)

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
        self.data["ruptures"] = self.data.pop("ruptures")
        site_parameters = self.data.pop("site-parameters", {})

        for site_key in site_parameters:
            try:
                site_parameters[site_key] = float(site_parameters[site_key])
            except ValueError:
                site_parameters[site_key] = site_parameters[site_key]

        for i, rupture in enumerate(self.data["ruptures"]):
            if "z_tor" in rupture.keys() and "rrup" not in rupture.keys():
                self.data["ruptures"][i]['rrup'] = (
                    rupture['rjb'] ** 2 + rupture["z_tor"] ** 2) ** 0.5
            if "rake" not in rupture.keys():
                self.data["ruptures"][i]["rake"] = 0.0

            if not self.USE_FULL_PSHA:
                self.data["ruptures"][i]["gmms"] = np.arange(
                    len(self.data["gmms"])).tolist()

            self.data["ruptures"][i] = {
                **self.data["ruptures"][i],
                **site_parameters
            }

    def _ensure_total_weights_length(self, counter, weights):
        # Ensure that total number of scenarios for each IM1 matches
        # length of total weights
        for im, w in weights.items():
            n_sc = counter[im]
            if len(w) != n_sc:
                raise ValueError(
                    f"Total number of scenarios {n_sc} not matching"
                    f" length of total-weights ({len(w)}) for IM1 {im}"
                )

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
            "gmms", "ruptures", "num_components",
            "component_definition", "total_weights",
            "add_data_for_dis", "avg_sa",
        ]

        data = {}
        for key, value in self.data.items():
            key = key.replace("-", "_")
            if key in create_vars:
                data[key] = value
        return data

    def create(self):
        return self._create(**self._create_input)

    def _create(
        self,
        gmms: dict,
        ruptures: dict,
        total_weights: dict,
        num_components: int,
        component_definition: str,
        add_data_for_dis: dict = None,
        avg_sa: dict = None,
    ):
        num_components = int(num_components)

        self.output_create["num_components"] = num_components
        if num_components == 1:
            component_definition = "RotD50"

        self.output_create["component_definition"] = component_definition

        if avg_sa is not None:
            self.im2, self.im1, gmms, total_weights = \
                self._identify_indirect_sa_avg(
                    avg_sa, self.im2, self.im1, gmms, total_weights
                )

        im2_range = np.asarray(self.im2['value'])
        im2_probs = np.zeros(im2_range.shape)

        scenarios, imi, mu_rup, sigma_rup, cov_rup, gmms_dict = \
            self._initialize_create(
                ruptures, [self.im1['name']], gmms, total_weights)

        mu_imstar_rup = {}
        sigma_imstar_rup = {}
        # cov_imstar_rup = {}
        for _im, arr in mu_rup.items():
            mu_imstar_rup[_im] = np.zeros(arr.shape)

            # Add third dimension to store for each value of im2
            mu_imstar_rup[_im] = np.tile(
                mu_imstar_rup[_im][:, :, np.newaxis],
                (1, 1, self.im2['num_pts'])
            )

            sigma_imstar_rup[_im] = np.zeros(sigma_rup[_im].shape)
            # cov_imstar_rup[_im] = np.zeros(cov_rup[_im].shape)

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
        for i, _rupture in enumerate(ruptures):
            mu_im_star[i], sigma_im_star[i], weights_im_star[i] = \
                self.get_conditional_im(
                self.im2, _rupture, gmms, component_definition,
                num_components)

        # Back-calculate epsilon at the IM2
        # Shape = (len(ruptures), len(im_star_gmms))
        # keys: GMM_name(s) and 'ws' in case of disaggregation
        imtls = self.dis_oq["imt"]
        curves = self.dis_oq["hazard-curves"]

        im_list = list(imtls.keys())
        im2_idx = im_list.index(self.im2['name'])

        poe_list = curves[:, 0, im2_idx, :].squeeze()
        im_list = imtls[self.im2['name']]

        interpolator = interp1d(im_list, poe_list, fill_value="extrapolate")
        im2_poes = interpolator(im2_range).squeeze()

        im_star_gmms = set()
        eps_im_star['ws'] = {}
        for i, _mu_im_star in mu_im_star.items():
            gmm_ids = ruptures[i]['gmms']
            rup_gmms = self._find_gmm_by_key(gmms, self.im2['type'], gmm_ids)

            im_star_gmms = im_star_gmms.union(set(_mu_im_star.keys()))
            eps_im_star[i] = {}
            for g, _mu in _mu_im_star.items():
                eps_im_star[i][g] = calculate_epsilon(
                    im2_range, _mu, sigma_im_star[i][g])

                if self.USE_FULL_PSHA:
                    _g_idx = rup_gmms['names'].index(g)

                    # Logic tree leaf weight
                    lt_weight = rup_gmms['weights'][_g_idx]

                    _poe = self._truncnorm_sf(
                        add_data_for_dis['phi_b'], eps_im_star[i][g]
                    )
                    ws = -np.log((1 - ruptures[i]['probs']) ** _poe) / \
                        add_data_for_dis['inv_time']

                    # To avoid divide by zero
                    # convert 0 poes to negligible value
                    im2_poes[im2_poes < self.NEGLIGIBLE] = self.NEGLIGIBLE

                    ws /= -np.log(1 - im2_poes) / \
                        add_data_for_dis['inv_time']
                    _ws = ws * lt_weight

                    # This does not impact the results
                    # IMs that have not been present during PSHA,
                    # won't have weights equal to 1
                    _ws[_ws < self.NEGLIGIBLE] = self.NEGLIGIBLE

                    if g in eps_im_star['ws']:
                        eps_im_star['ws'][g].append(_ws)
                    else:
                        eps_im_star['ws'][g] = [_ws]

        if self.USE_FULL_PSHA:
            # Normalize weights to sum to 1 for each IM2 value
            for g in eps_im_star['ws'].keys():
                eps_im_star['ws'][g] = np.asarray(eps_im_star['ws'][g])
                sum_along_axis0 = np.sum(eps_im_star['ws'][g], axis=0,
                                         keepdims=True)
                sum_along_axis0[sum_along_axis0 < self.NEGLIGIBLE] = \
                    self.NEGLIGIBLE
                eps_im_star['ws'][g] = eps_im_star['ws'][g] / sum_along_axis0

        # Used in the case where GMM does not apply to IM2
        # Not sure about this... this is an approximation for convenience
        # No reference for it
        # In the case of usage of diassgregation results,
        # this will never be used
        for i, _rupture in enumerate(ruptures):
            eps_im_star_comb[i] = self._calculate_epsilon_for_rup(
                self.im2["value"], mu_im_star[i], sigma_im_star[i],
                weights_im_star[i]
            )

        # For each rupture scenario
        im = self.im1['type']

        if self.correlation_type == "high":
            for _im in rho_imi_im_star.keys():
                rho_imi_im_star[_im] = rho_imi_im_star[_im] ** (1 / 2)

        elif self.correlation_type == "one":
            for _im in rho_imi_im_star.keys():
                rho_imi_im_star[_im] = np.array([0.99])

        # IMi calculations (IM1 calculations)
        for i, scenario in enumerate(scenarios[im]):

            # Retrieve epsilon
            if scenario["gmm"] in im_star_gmms:
                epsilon = eps_im_star[scenario["rup_id"]][scenario["gmm"]]
            else:
                epsilon = eps_im_star_comb[scenario["rup_id"]]

            scenario["im_name"] = im

            # Estimate means and stddevs conditioned
            # on rupture scenario, \mu_{ln(IM1|rup)}, \sigma_{ln(IM1|rup)}
            mu_rup[im][:, i], sigma_rup[im][:, i] = \
                self._get_all_means_stds(
                    imi[im], scenario, component_definition,
                    num_components)

            # Compute \sigma_{ln(IM1|IM2,rup)}
            sigma_imstar_rup[im][:, i] = get_sigma_xy(
                sigma_rup[im][:, i], rho_imi_im_star[im])

            # Compute \mu_{ln(IM1|IM2,rup)}
            # Initially compute for each GMM case of IM2
            # shape=(len(gmms_IM2), len(im1[im]))
            mu_imstar_rup[im][:, i] = get_mean_xy(
                mu_rup[im][:, i], sigma_rup[im][:, i], rho_imi_im_star[im],
                epsilon)

        # Logarithmic median of target spectrum accounting for
        # all cases (GMMs and rupture scenarios) through contribution factors
        # Exact approach
        weights_imi = {}
        for im_type, _ in mu_imstar_rup.items():
            if self.USE_FULL_PSHA:
                # Weights from occurrence rates of ruptures
                # For a single GMM
                weights = self._get_weight_disaggregation(
                    eps_im_star, gmms_dict[im_type])

            else:
                # Logic tree weights
                weights = self._get_par_from_scenarios("w", scenarios[im_type])

            weights_imi[im_type] = weights

        # Site-specific probability density function is computed using
        # mu_imstar_rup = mu_{lnIM1|lnIM2,rup}, shape=(1, n_rups, im2.num_pts)
        # sigma_imstar_rup = sigma_{lnIM1|lnIM2,rup}, shape=(1, n_rups)
        im_type = self.im1['type']
        sigma = np.squeeze(sigma_imstar_rup[im_type], axis=0)
        mu = np.squeeze(np.exp(mu_imstar_rup[im_type]), axis=0)

        im1_range, im1_probs, pdfs = self._get_im1_cdf(sigma, mu)

        # Compute pff in matrix form (element-wise multiplication)
        # im1_probs: (K,) -> broadcast to (L, 1, 1)
        # pdfs: (K, M, N)
        # weights_imi[im_type]: (M, N)
        # pff: (K, M, N)
        pff = im1_probs[:, None, None] * pdfs * weights_imi[im_type]
        # Integrate using Simpson's rule along im1_range axis (axis=0) to
        # reduce to (M, N)
        integrated_pff = simpson(pff, x=im1_range, axis=0)  # (M, N)
        # Sum over M axis to get im2_probs (N,)
        im2_probs = np.sum(integrated_pff, axis=0)  # (N,)

        # ------------------
        # DEV plotters
        # cdfs = lognorm.cdf(im1_range[:, None, None], sigma[:, None], 0, mu)

        # import matplotlib.pyplot as plt

        # print(mu[:, 30])
        # print(self.im2['value'][30])

        # plt.plot(im1_range, pdfs[:, :, 30], c="grey")
        # plt.plot(im1_range, cdfs[:, :, 30], c="g")
        # plt.plot(im1_range, im1_probs, c="r")
        # print(integrated_pff.shape)
        # print(im1_range.shape)

        # plt.plot(im2_range, im2_probs, c="k", ls="--")

        # plt.xlim(0, 5)
        # plt.ylim(0, 10)
        # plt.show()

        # ------------------
        # # KEPT TO REMEMBER THE PROCEDURE
        # for i, _ in enumerate(im2_range):
        #     for j, (_mu, _sigma) in enumerate(zip(mu[:, i], sigma)):
        #         pff = (
        #             im1_probs
        #             * lognorm.pdf(
        #                 im1_range, _sigma, 0, _mu
        #             )
        #             * weights_imi[im_type][j, i]
        #         )

        #         im2_probs[i] += simpson(pff, x=im1_range)

        # -------
        # Ensure probabilities do not decrease because of
        # missing computational data at very high IMs
        im2_probs = np.maximum.accumulate(im2_probs)

        # Limit maximum values to 1
        # a problem can occur for low DS where im range does not capture
        # many data points at lower intensity values
        im2_probs = np.clip(im2_probs, None, 1)

        return im2_probs, im2_range

    def _get_im1_cdf(self, sigma, mu):

        im1_range = self.im1.get("im_range")
        im1_probs = self.im1.get("probs")

        _max_limit = 50
        if self.im1["type"] in _IMS_VEL:
            _max_limit = 2000

        if im1_range is None or im1_probs is None:
            # fragility function for IM1
            pc_min = 0.0  # I'd say set this to 0
            pc_max = np.nextafter(1.0, 0)  # Consider the max value after 1.0
            pc_max = np.clip(pc_max, 0, 0.9999)

            self.im1['min'] = pc_min
            self.im1['num_pts'] = 500
            # Initial guess for the upper bound
            # Get the max value of all possible sampled medians
            # for im1: np.max(mu)
            # Find the row index where the maximum value in mu occurs

            # Select the corresponding value of sigma
            if sigma.ndim != 1:
                _max_idx = np.unravel_index(np.argmax(mu), mu.shape)
                _sigma_max = sigma[_max_idx]
            else:
                # Find the row index where the maximum value in mu occurs
                _max_idx = np.unravel_index(np.argmax(mu), mu.shape)[0]
                _sigma_max = sigma[_max_idx]
            _mu_max = np.max(mu)

            self.im1['max'] = max(15, lognorm.ppf(
                pc_max, _sigma_max, loc=0, scale=_mu_max
            ))

            # Ensure it is not too high
            self.im1['max'] = min(self.im1['max'], _max_limit)

            # Increase to avoid truncation of PDF values and lowering the CDF
            # Set the lower bound to 0, and increase upper bound iteratively
            factor = 1  # Initial factor
            MAX_ITER = 1  # Maximum number of iterations to broaden im1_range
            THRESHOLD = 1e-3  # Threshold for pdf check

            # import tracemalloc
            # tracemalloc.start()

            for i in range(MAX_ITER):
                if self.im1["type"] in _IMS_ACC:
                    _first_part_n = int(0.75 * self.im1['num_pts'])
                    first_part = np.linspace(
                        0, 2, _first_part_n, endpoint=False)

                    # Remaining points from 10 to max
                    second_part = np.linspace(
                        2, self.im1['max'],
                        self.im1['num_pts'] - _first_part_n)

                    # Combine them
                    im1_range = np.concatenate([first_part, second_part])
                else:
                    im1_range = np.linspace(
                        0, self.im1['max'], self.im1['num_pts'])

                # Compute the lognormal PDFs in a fully vectorized way
                # im1_range: (K,) -> broadcast to (K, M, N)
                # sigma: (M,) -> reshaped to (M, 1) for broadcasting
                # mu: (M, N) remains unchanged
                # pdfs = lognorm.pdf(
                #     im1_range[:, None, None], sigma[:, None], 0, mu)

                pdfs = np.empty((len(im1_range), len(sigma), mu.shape[1]),
                                dtype=np.float32)
                _sigma = sigma[:, None] if sigma.ndim == 1 else sigma
                for i, _im1 in enumerate(im1_range):
                    pdfs[i] = lognorm.pdf(_im1, _sigma, 0, mu)

                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Current memory usage: {current / (1024 * 1024):.2f} "
                #       "MB")
                # print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")

                # Check boundary conditions (right and left ends of im1_range)
                too_small_right = np.any(pdfs[-1, :, :] > THRESHOLD)
                too_large_left = np.any(pdfs[0, :, :] > THRESHOLD)
                if too_small_right and i + 1 < MAX_ITER:
                    # We need to check this iteratively
                    print("Increase the right side of im1_range")
                    factor += 1
                    self.im1['num_pts'] = int(self.im1['num_pts'] * factor)
                    self.im1['max'] = self.im1['max'] * factor
                if too_large_left:
                    # This should not happen for lower bound 0.0
                    print("Decrease the left side of im1_range")
                if not too_small_right and not too_large_left:
                    break

            self.im1['max'] = im1_range[-1]

            # CDF for IM1
            im1_probs = lognorm.cdf(
                im1_range, self.im1['dispersion'], 0, self.im1['median']
            )

        else:
            im1_range = np.asarray(im1_range)
            im1_probs = np.asarray(im1_probs)

            # Ensure that it starts with 0.0
            if im1_range[0] == 0 and im1_probs[0] != 0:
                im1_probs[0] = 0.0
            if not (im1_range[0] == 0 and im1_probs[0] == 0):
                im1_range = np.insert(im1_range, 0, 0.0)
                im1_probs = np.insert(im1_probs, 0, 0.0)

            max_limit = 50
            min_limit = 0.001
            if self.im1["type"] in _IMS_VEL:
                min_limit = 1
                max_limit = 2000

            new_im1_range = np.geomspace(min_limit, max_limit, 500)
            new_im1_range = np.insert(new_im1_range, 0, 0.0)
            f = interp1d(
                im1_range,
                im1_probs,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False
            )
            new_im1_probs = f(new_im1_range)
            new_im1_probs = np.clip(new_im1_probs, 0, 1)
            im1_range = new_im1_range
            im1_probs = new_im1_probs

            pdfs = np.empty((len(im1_range), len(sigma), mu.shape[1]),
                            dtype=np.float32)
            _sigma = sigma[:, None] if sigma.ndim == 1 else sigma
            for i, _im1 in enumerate(im1_range):
                pdfs[i] = lognorm.pdf(_im1, _sigma, 0, mu)

        return im1_range, im1_probs, pdfs
