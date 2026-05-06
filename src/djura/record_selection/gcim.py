# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from pathlib import Path
import yaml
import json
import numpy as np
from typing import Union, List

from ._gcim_conditional import _GCIMConditional
from ._gcim_unconditional import _GCIMUnconditional
from .utilities import get_list_id, get_periods_ims, get_period_im
from .constants import DB_CAUSAL_PARS, SUPPORTED_IM_DESCRIPTORS, \
    CORRELATION_MODELS, SUPPORTED_IM_COMPONENTS
from ..data_loader import get_nga_west2


class GCIM:
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
            # Ignore records if key is matched in metdata
            "EQ_name": None,
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
        data: Union[Path, str, dict] = None,
        conditional: bool = None,
        records: dict = None,
        dis_oq: dict = None,
        poe_for_selection: float = None,
    ) -> None:
        """Perform Conditional-based or Unconditional-based
        ground motion selection based on
        Generalized Conditional Intensity Measure (GCIM) approach

        Parameters
        ----------
        data : Union[Path, str, dict], optional
            File containing information on input arguments,
            Can concern both create() and select() methods,
            Can be a dictionary, "*.json", or "*.yaml"
            by default self.data
        conditional : bool, optional
            True for conditional IM-based selection,
            otherwise unconditional-based,
            if None will infer from data input, i.e., "im-star"
            by default None
        records : dict, optional
            Required for plotter only
            Output of GCIM selector
        dis_oq : dict, optional
            Disaggregation data from OQ engine,
            Obtained through get_context_from_dstore() function
            Overrides:
                gmms; site-parameters; ruptures; imi; im-star
            by default None
        poe_for_selection : float, optional
            Needed only in combination with disaggregation data
            by default, None
        """

        self.USE_DISAGGREGATION_DATA = False

        # Data
        self.data: dict = None

        # Output of .create() method
        self.output_create: dict = None

        # Selected records, output of .select() method
        self.records: dict = {
            "selected_scaled_best": None,
            "selected_scaled_total": None,
        }

        # Conditional or Unconditional GCIM selection
        self._parent: Union[_GCIMConditional, _GCIMUnconditional] = \
            _GCIMUnconditional(None)

        self.metadata = get_nga_west2()

        if records is not None:
            self.records = records

        self.conditional = conditional

        # Validate input parameters
        if data is not None:
            self._read_input_file(data)
            self._set_parent()

        # OQ Engine disaggregation data
        if dis_oq is not None:
            self._read_from_oq_dis(dis_oq, poe_for_selection)
            self.USE_DISAGGREGATION_DATA = True
            self._set_parent()

    def _read_from_oq_dis(self, dis_oq, poe_for_selection):
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
            IM* name, value will be inferred from hazard curves
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
        if poe_for_selection is None:
            raise ValueError("poe_for_selection is missing")

        poe_idx = None
        try:
            poe_idx = [dis_oq['oq-poes'].index(poe_for_selection)]
        except ValueError:
            poes = dis_oq['oq-poes']
            higher = min(
                (p for p in poes if p > poe_for_selection), default=None)
            lower = max(
                (p for p in poes if p < poe_for_selection), default=None)

            if higher is not None and lower is not None:
                higher_idx = poes.index(higher)
                lower_idx = poes.index(lower)
                poe_idx = [higher_idx, lower_idx]
            else:
                raise ValueError(
                    f"{poe_for_selection} is not in the list of available"
                    " POEs from OQ disaggregation data.")

        # IM types
        imtls = dis_oq["imt"]

        # IMLs
        self.data['imi'] = list(dis_oq['imt'].keys())

        # IM*
        if self.conditional and dis_oq['im_ref'] is not None:
            from .gsim.oq import compute_hazard_maps
            curves = dis_oq["hazard-curves"]

            try:
                im_star_idx = self.data['imi'].index(dis_oq['im_ref'])
            except KeyError:
                raise ValueError(
                    f"IM* {dis_oq['im_ref']} not in the list of"
                    f"defined IMs: {self.data['imi']}")

            # Hazard curves (imls.shape=(1, len(poes)))
            imls = compute_hazard_maps(
                curves[:, 0, im_star_idx, :], imtls[dis_oq['im_ref']],
                dis_oq['oq-poes'])

            self.data['im-star'] = {
                "type": dis_oq['im_ref'],
                "value": np.mean(imls[0][poe_idx]),
            }

        # Context parameters
        ctx_by_grp = dis_oq['ctx_by_grp']

        # Site parameters
        site_params = {}
        # Based on OQ assignment
        first_key = next(iter(ctx_by_grp))
        for param in dis_oq['site-parameters']:
            site_params[param] = ctx_by_grp[first_key][param][0]

        self.data['site-parameters'] = site_params

        # GMMs and logic tree leaf weights
        # lt_weight = gmm_weight * source_model_weight
        lt_weights = dis_oq['lt-weights']
        _imts, _ = get_periods_ims(imtls)
        _imts = set(_imts)
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

            # TODO: There could be an issue here
            # src_gmm_weights = [lt_weights[ids] for ids in weight_ids][-1]
            src_gmm_weights = lt_weights[weight_ids].reshape(1, -1)[-1]

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
            "poe_for_selection": poe_for_selection,
        }

    def _set_parent(self):
        if self.conditional is None:
            self.conditional = self._conditional_or_not()

        if self.conditional:
            self._parent = _GCIMConditional(
                self.metadata, self.USE_DISAGGREGATION_DATA)
        else:
            self._parent = _GCIMUnconditional(self.metadata)

    def _conditional_or_not(self) -> bool:
        """Check whether performing conditional selection or not
        based on "im-star" information

        Returns
        -------
        bool
            True for Conditional; False for Unconditional
        """
        if "im-star" not in self.data:
            return False
        if self.data["im-star"] is None:
            return False
        return True

    def create(
        self,
        data: Union[Path, str, dict] = None,
    ) -> dict:
        """Creates the target GCIM distribution (conditional or unconditional)

        Notes
        ----------
        https://docs.openquake.org/oq-engine/master/openquake.hazardlib.gsim.html
        in order to check required input parameters for the ground motion
        models, e.g. rupture parameters (rup_param), site parameters
        (site_param), distance parameters (dist_param).
        Rupture parameters 'fhw', 'azimuth', 'upper_sd' and 'lower_sd' are
        used to derive some gmm parameters
        in accordance with Kaklamanos et al. 2011 within
        ConditionalSpectrum._set_contexts method. They are not required
        by any gmm.

        References
        ----------
        Bradley, B.A. (2010). A generalized conditional intensity measure
        approach and holistic ground-motion selection.
        Earthquake Engineering & Structural Dynamics, 39. DOI: 10.1002/eqe.995

        Bradley, B.A. (2012). A ground motion selection algorithm based on the
        generalized conditional intensity measure approach.
        Soil Dynamics and Earthquake Engineering, 40, 48-61.
        https://doi.org/10.1016/j.soildyn.2012.04.007

        Lin, T., Harmsen, S. C., Baker, J. W., & Luco, N. (2013).
        Conditional Spectrum Computation Incorporating Multiple Causal
        Earthquakes and Ground-Motion Prediction Models. In Bulletin of the
        Seismological Society of America (Vol. 103, Issue 2A, pp. 1103-1116).
        https://doi.org/10.1785/0120110293

        Tarbali, K., & Bradley, B.A. (2015). Ground motion selection for
        scenario ruptures using the generalised conditional intensity measure
        (GCIM) method. Earthquake Engineering & Structural Dynamics, 44,
        1601 - 1621. DOI: 10.1002/eqe.2546

        Baker, J.W., & Lee, C.B. (2018). An Improved Algorithm for Selecting
        Ground Motions to Match a Conditional Spectrum.
        Journal of Earthquake Engineering, 22, 708 - 723.
        DOI:10.1080/13632469.2016.1264334

        Parameters
        ----------
        data : Union[Path, str, dict]
            File containing information on input arguments
            Required, if 'data' was not provided for GCIM object
            Overrides global 'self.data', by default None

        **Required parameters of** ``data``

        ruptures : List[dict]
            Rupture scenarios with hazard context parameters and
            GMM associations. Example::

                [
                    {"mag": float, "weight": Optional[float],
                                   "gmms": Optional[ID]},
                    {"mag": 5,     "weight": Optional[float],
                                   "gmms": Optional[ID]},
                ]

            ``mag`` is one example context parameter; ``weight`` is the rupture
            weight (sum must be 1.0). If omitted, provide ``total-weights``
            under the ``gmms`` key instead.
        gmms : List[dict]
            Ground motion models associated with each IM type.
            The weights are optional; if omitted, provide total weights under
            the ``weights`` sub-key of each ``ruptures`` entry. Example::

                [
                    {"ID": Optional[int],
                     im1: {"names": List[str],
                           "weights": Optional[List],
                           "total-weights": Optional[List]},
                     im2: {...}},
                    {"ID": Optional[int], ...}
                ]

            If ``ID`` is not provided it is inferred from the list index.
            The sum of ``weights`` (and of ``total-weights``) must be 1.0
            per IM type.
            ``total-weights`` = ``gmms.weights`` × ``ruptures.weights``

        **Required context parameters**

        vs30: float
            Average shear-wave velocity of the site, [m/s]
        mag : float
            Magnitude of the earthquake (required by all gmm)
        rjb : float
            Closest distance to surface projection of coseismic rupture [km]

        **Optional context parameters**

        vs30measured : bool
            vs30 type, True (measured) or False (inferred)
        z1pt0 : float
            Depth to Vs=1 km/sec from the site
        z2pt5 : float
            Depth to Vs=2.5 km/sec from the site, in [km]
        rake : float
            Fault rake
        dip : float
            Fault dip
        width : float
            Fault width
        hypo_depth : float
            Hypocentral depth of the rupture, [km]
        ztor : float
            Depth to top of coseismic rupture [km]
        fhw : int
            Hanging-wall factor,
            1 for site on down-dip side of top of rupture;
            0 otherwise
        azimuth : float
            Source-to-site azimuth, alternative of hanging wall factor
        upper_sd : float
            Upper seismogenic depth
        lower_sd : float
            Lower seismogenic depth
        rrup : float
            Closest distance to coseismic rupture [km]
        repi : float
            Epicentral distance [km]
        rhypo : float
            Hypocentral distance [km]
        rx : float
            Horizontal distance from top of rupture measured perpendicular
            to fault strike [km]
        ry0 : float
            The horizontal distance off the end of the rupture measured
            parallel to strike [km]
        z_tor : float
            Depth to the top of the rupture plane, by default 1

        **Optional parameters of** ``data``

        num-components : int, optional
            1 for single-component selection and arbitrary component sigma.
            2 for two-component selection and average component sigma,
            by default 2
        component-definition : str, optional
            The spectra definition, 'GeoMean', 'RotD50', 'RotD100'. Necessary
            if num-components = 2, by default 'RotD50'
        imi : List[str], optional
            IMis to be used for GCIM distribution creation. Default::

                ['SA(0.05s)', 'SA(0.075s)', 'SA(0.1s)', 'SA(0.15s)',
                 'SA(0.2s)', 'SA(0.25s)', 'SA(0.3s)', 'SA(0.4s)',
                 'SA(0.5s)', 'SA(0.75s)', 'SA(1.0s)', 'SA(1.3s)',
                 'SA(1.5s)', 'SA(2.0s)']

        **Required parameters of** ``data`` **(conditional GCIM only)**

        im-star : dict
            Conditioning IM descriptor. Keys: ``type`` (IM type, e.g.
            ``'SA'``), ``value`` (conditioning level), ``period``
            (conditioning period in seconds; not required for
            period-independent IMs). Example::

                {"type": str, "value": float, "period": Optional[float]}

            If ``None``, the target is an unconditional spectrum;
            If not None, target is conditional spectrum unless overriden by
            **self.conditional** parameter;

        Returns
        ----------
        dict
            Dictionary containing the GCIM distribution and
            key meta information, the keys are described as follows

        'im-star' : dict
            Conditional IM descriptors, same as input "im-star"
        'target' : dict
            Target multivariate GCIM distribution. Keys:

            - ``mu_lnIMi``: mean for all rupture scenarios and GMMs
            - ``sigma_lnIMi``: stdv for all rupture scenarios and GMMs
            - ``cov_lnIMi``: covariance matrix for all rupture scenarios
              and GMMs
            - ``IMi``: ground motion intensity measures (IMs)
            - ``correlations``: correlation matrices between all IMi types

        'data' : dict
            Extra information during intermediate calculations
            mu_lnIMi_rup, sigma_lnIMi_rup, cov_lnIMi_rup, mu_lnIMj_rup,
            sigma_lnIMj_rup, epsilon_lnIMj_rup, mu_lnIMi_lnIMj_rup,
            sigma_lnIMi_lnIMj_rup, cov_lnIMi_lnIMj_rup, case_weights
        """
        # Read and validate input parameters
        if data is not None:
            self._read_input_file(data)
            self._set_parent()

        self._validate_create_input()

        # Pop irrelevant arguments and prepare input
        data = self._pop_keys(method="create")

        # Create the GCIM distributions
        self._parent.create(**data)

        self.output_create = self._parent.output_create

        return self.output_create

    def select(
        self,
        data: Union[Path, str, dict] = None,
        output_create: Union[Path, str, dict] = None,
    ) -> dict:
        """Perform the ground motion selection

        Parameters
        ----------
        data : Union[Path, str, dict], optional
            File containing information on input arguments,
            Required, if 'data' was not provided for GCIM object
            Overrides global 'self.data', by default None
        output_create : dict, optional
            Outputs of 'create' method
            Required if run with no previous run of 'create'
            by default None

        **Parameters of** ``data``

        nrun : int, optional
            Number of separate runs, by default 1
        nreplicate : int, optional
            Number of replicates, by default 1
            The algorithm is repeated for nreplicate times
        num_records : int, optional
            Number of ground motions to be selected, by default 40
        seed : int, optional
            For repeatability. For a particular seed not equal to
            zero, the code will output the same set of ground motions.
            The set will change when the 'seed' value changes. If set to
            zero, the code randomizes the algorithm and different sets of
            ground motions (satisfying the target mean and variance) are
            generated each time, by default 0
        ks_alpha : float, optional
            Kolmogorov-Smirnov test significance level, by default 0.05
        im_weights : List[float], optional
            Weights of IMs, must match the number of items under self.data.imi
            by default 1.0 for each IM type
        context_limits : dict, optional
            Limiting values on context parameters; keys must be present
            in the metadata. By default ``None``. Example::

                {"magnitude": [6, 7]}  # events of magnitude 6–7 only
        max_scaling_factor : float, optional
            Maximum scaling factor allowed, by default 1,
            i.e. no scaling allowed

        Returns
        ----------
        dict
            Selected record information
        """
        if data is not None:
            self._read_input_file(data)
            self._set_parent()

        self._validate_create_input()

        if output_create is not None:
            # Update for the parent as well
            self.output_create = output_create
            self._parent.output_create = output_create

        self._validate_select_input()

        # Pop irrelevant arguments
        data = self._pop_keys(method="select")

        self._parent.select(**data)

        self.records = {
            "selected_scaled_best": self._parent.selected_scaled_best,
            "selected_scaled_total": self._parent.selected_scaled_total,
        }

        return self.records

    def _pop_keys(self, method: str) -> dict:
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
            "gmms", "ruptures", "imi", "num_components",
            "component_definition", "total_weights",
        ]

        cond_create_vars = [
            "im_star", "gmms", "ruptures", "imi", "num_components",
            "component_definition", "total_weights", "add_data_for_dis",
            "avg_sa",
        ]

        select_vars = [
            "nreplicate", "num_records", "context_limits", "seed",
            "ks_alpha", "im_weights", "max_scaling_factor", "greedy_loops"
        ]

        if isinstance(self._parent, _GCIMConditional):
            create_vars = cond_create_vars

        data = {}
        for key, value in self.data.items():
            key = key.replace("-", "_")
            if method == "create":
                if key in create_vars:
                    data[key] = value
            else:
                if key in select_vars:
                    data[key] = value
        return data

    def _validate_select_input(self):
        """Validate input arguments for select method

        Raises
        ------
        ValueError
            If length of 'imi' and 'im_weights' do not match
        """
        # Infer self.data parameters based on create() output
        # to avoid using the defaults if data was not provided
        self.data["component_definition"] = \
            self.output_create["component_definition"]
        self.data["num_components"] = self.output_create["num_components"]

        imi = []
        for im_type, periods in self.output_create["target"]["IMi"].items():
            if len(periods) == 0:
                imi.append(im_type)
                continue
            for t_i in periods:
                imi.append(f"{im_type}({t_i}s)")
        self.data["imi"] = imi

        if "im-star" in self.output_create and \
                self.output_create["im-star"] is not None:
            self.data["im-star"] = self.output_create["im-star"]

        # If weights are missing, use default values, i.e.
        # equal weight for each IM type
        if len(self.data["im_weights"]) == 0 or "im_weights" not in self.data \
                or self.data["im_weights"] is None:
            self.data["im_weights"] = np.ones(len(self.data["imi"]))

        if len(self.data["imi"]) != len(self.data["im_weights"]):
            raise ValueError("Length of 'imi' and 'im_weights' must match")

    def _validate_create_input(self):
        """Validate input arguments for create method
        """
        if "component-definition" in self.data:
            if self.data["component-definition"] == "rotd50":
                self.data["component-definition"] = "RotD50"
            if self.data["component-definition"] == "rotd100":
                self.data["component-definition"] = "RotD100"

        # Add missing IMi to metadata
        imi_types, imi_periods = get_periods_ims(self.data["imi"])

        for im_i, t_i in zip(imi_types, imi_periods):
            if im_i == "Sa_avg":
                continue

            self._parent._add_missing_im(im_i, t_i)

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
                            self._parent._validate_gmm_indirect_sa_avg(
                                self._parent._validate_gmm(name, **kwarg),
                                **avg_sa_kwarg
                            ))
                    else:
                        bgmpes.append(
                            self._parent._validate_gmm(name, **kwarg))

                gmms[im]["bgmpe"] = bgmpes
                gmms[im]["names"] = src_gmms

            if "ID" not in gmms:
                gmms["ID"] = _id

        # Conditional IM (IM*)
        im_star = self.data['im-star']

        if im_star is not None and self.conditional:
            im_type, period = get_period_im(im_star["type"])
            im_star_value = im_star["value"]

            im_star = {
                "type": im_type,
                "value": im_star_value,
                "period": period,
            }

            # Add missing IM* to the metadata
            if "period" in im_star and im_type != "Sa_avg":
                self._parent._add_missing_im(
                    im_star["type"], im_star["period"]
                )

        self.data["im-star"] = im_star

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

            if not self.USE_DISAGGREGATION_DATA:
                self.data["ruptures"][i]["gmms"] = np.arange(
                    len(self.data["gmms"])).tolist()

            self.data["ruptures"][i] = {
                **self.data["ruptures"][i],
                **site_parameters
            }

        # In case of True, Weights of ruptures will be inferred from OQ
        # results during target creation
        if not self.USE_DISAGGREGATION_DATA:
            # Ensure the weights of cases associated with each GMM equal to 1.0
            self.data["total-weights"] = self._validate_weights(
                self.data["ruptures"], self.data["gmms"])

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
            Total hazard contributions per IMi
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

        if total_weights is not None:
            for im, w in total_weights.items():

                total_weights[im] = w / np.sum(w)

            self._ensure_total_weights_length(scenario_counter,
                                              total_weights)

            return total_weights

        # Hazard contributions (total-weights) not provided
        # inferring from logic tree of ruptures and GMMs for each IMi
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

    def _ensure_total_weights_length(self, counter, weights):
        # Ensure that total number of scenarios for each IMi matches
        # length of total weights
        for im, w in weights.items():
            n_sc = counter[im]
            if len(w) != n_sc:
                raise ValueError(
                    f"Total number of scenarios {n_sc} not matching"
                    f" length of total-weights ({len(w)}) for IMi {im}"
                )

    def get_all_scenarios(self):

        # Validate input parameters
        self._validate_create_input()

        ruptures = self.data["ruptures"]
        imi = self.data["imi"]
        gmms = self.data["gmms"]
        total_weights = self.data["total-weights"]

        all_scenarios, _, _, _, _, _ = self._parent._initialize_create(
            ruptures, imi, gmms, total_weights
        )
        return all_scenarios

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

    def get_supported_rupture_parameters(self) -> frozenset:
        """Gets supported rupture parameters

        Returns
        -------
        set
            Names of rupture parameters supported
        """
        return self._parent._get_supported_parameters("rupture")

    def get_supported_sites_parameters(self) -> frozenset:
        """Gets supported sites parameters

        Returns
        -------
        set
            Names of sites parameters supported
        """
        return self._parent._get_supported_parameters("sites")

    def get_supported_distances_parameters(self) -> dict:
        """Gets supported distances parameters

        Returns
        -------
        set
            Names of distances parameters supported
        """
        return self._parent._get_supported_parameters("distances")

    def get_supported_ims(self) -> dict:
        return SUPPORTED_IM_DESCRIPTORS

    def get_metadata_parameters(self) -> set:
        return set(self.metadata.keys())

    def available_correlation_models(self) -> dict:
        return CORRELATION_MODELS

    def get_supported_im_component_types(self) -> dict:
        return SUPPORTED_IM_COMPONENTS

    def get_available_gsims(self):
        return self._parent.get_available_gsims()

    def check_gmpe_attributes(self, gmpe: str):
        self._parent.check_gmpe_attributes(gmpe)

    def get_gmpe_parameters(self, gmpe: str):
        return self._parent.get_gmpe_attributes(gmpe)

    def get_causal_pars_db(self):
        # self.metadata.keys()

        return DB_CAUSAL_PARS
