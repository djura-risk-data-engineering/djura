# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ./NOTICE.md for full attribution.

from typing import Tuple
import numpy as np

from .const import site_param_dt, KNOWN_DISTANCES, RUPTURE_PARAMETERS
from . import models as gsim_models
from .. import models as gmm_pydantic_models
from . import contexts
from .contexts import SitesContext, RuptureContext, \
    DistancesContext, Context
from ..utilities import inspect_file_for_classes


class OQ:
    MODEL_ATTRIBUTES = {
        "DEFINED_FOR_TECTONIC_REGION_TYPE": "Tectonic region type",
        "DEFINED_FOR_INTENSITY_MEASURE_TYPES": "Intensity measure types",
        "DEFINED_FOR_INTENSITY_MEASURE_COMPONENT":
        "Intensity measure component(s)",
        "DEFINED_FOR_STANDARD_DEVIATION_TYPES": "Standard deviation types",
        "REQUIRES_SITES_PARAMETERS": "Required site parameters",
        "REQUIRES_RUPTURE_PARAMETERS": "Required rupture parameters",
        "REQUIRES_DISTANCES": "Requires distance parameters",
        "SUGGESTED_LIMITS": "Recommended parameter limits",
    }

    def __init__(self) -> None:
        """Methods to prepare data
        """
        pass

    def _validate_gmm(self, gmm: str, **kwargs):
        """Ensure that GMPE input is available

        Parameters
        ----------
        gmm : str
            Ground motion model (GMM) name

        Returns
        -------
        .gsim._
            GMM model

        Raises
        ------
        KeyError
            gmm is not a valid gmpe name
        """
        gmm = gmm.split("_")[0]

        try:
            import inspect

            methods = inspect_file_for_classes(gsim_models)

            if gmm in methods:
                gmm_class = getattr(gsim_models, gmm)

                init_params = inspect.signature(gmm_class.__init__).parameters
                filtered_kwargs = {key: kwargs[key]
                                   for key in init_params if key in kwargs}

                # Validate kwargs
                gmm_models = inspect_file_for_classes(gmm_pydantic_models)
                model = f"{gmm}Model"

                if model in gmm_models:
                    model_class = getattr(gmm_pydantic_models, model)
                    filtered_kwargs = dict(
                        model_class.model_validate(filtered_kwargs))

                return gmm_class(**filtered_kwargs)
            else:
                raise KeyError(f'{gmm} is not a valid gmpe name"')
        except KeyError:
            raise KeyError(f'{gmm} is not a valid gmpe name"')

    def _validate_gmm_indirect_sa_avg(self, gmm: str, **kwargs):
        try:
            return gsim_models.GmpeIndirectAvgSA(gmm, **kwargs)
        except KeyError:
            raise KeyError(f'{gmm} is not a valid gmpe name"')

    def _set_sites_context(self, case: dict) -> SitesContext:
        """Sets sites calculation context for ground shaking intensity models

        Parameters
        ----------
        case : dict
            Rupture scenario data

        Returns
        -------
        SitesContext
            Sites calculation context for ground shaking intensity models
        """
        site_keys_defined = {'z1pt0', 'vs30', 'sids', 'vs30measured', 'z2pt5'}

        sctx = contexts.SitesContext(case)

        vs30 = case['vs30']

        if 'vs30measured' in case.keys():
            vs30measured = case['vs30measured']
        else:
            vs30measured = True

        if 'z1pt0' in case.keys():
            z1pt0 = case['z1pt0']
        else:
            z1pt0 = None

        if 'z2pt5' in case.keys():
            z2pt5 = case['z2pt5']
        else:
            z2pt5 = None

        if z1pt0 is None:
            # Chiou and Youngs 2014, eq. (1) - California and Non-Japan (m)
            z1pt0 = np.exp(
                -7.15 / 4 * np.log((vs30**4 + 571**4) / (1360**4 + 571**4)))
            # Chiou and Youngs 2014, eq. (2) - Japan (m)
            # z1pt0 = np.exp(
            #     -5.23/2 * np.log((vs30**2 + 412**2) / (1360**2 + 412**2)))

            # Eqns. from Kaklamanos et al. 2011 (m)
            # if "gmms" in case and 'ChiouYoungs' in case['gmm']:
            #     z1pt0 = np.exp(28.5 - 3.82 / 8
            #                    * np.log(vs30 ** 8 + 378.7 ** 8))
            # else:
            #     if vs30 < 180:
            #         z1pt0 = np.exp(6.745)
            #     elif 180 <= vs30 <= 500:
            #         z1pt0 = np.exp(6.745 - 1.35 * np.log(vs30 / 180))
            #     else:
            #         z1pt0 = np.exp(5.394 - 4.48 * np.log(vs30 / 500))

        if z2pt5 is None:
            # Campbell and Bozorgnia 2014, eq. (33) - California and Non-Japan
            z2pt5 = np.exp(7.089 - 1.144 * np.log(vs30)) * 1000  # km to m
            # # Campbell and Bozorgnia 2014, eq. (33) - Japan
            # z2pt5 = np.exp(5.359 - 1.102 * np.log(vs30)) * 1000  # km to m

            # Eqn. from Kaklamanos et al. 2011
            # z2pt5 = (519 + 3.595 * z1pt0)  # in meters

        # Site id
        setattr(sctx, 'sids', np.array([0], dtype='float64'))
        # Average shear-wave velocity of the site
        setattr(sctx, 'vs30', np.array([vs30], dtype='float64'))
        # vs30 type, True (measured) or False (inferred)
        setattr(sctx, 'vs30measured', np.array([vs30measured]))
        # Depth to Vs=1 km/sec (oq considers this in m)
        setattr(sctx, 'z1pt0', np.array([z1pt0], dtype='float64'))
        # Depth to Vs=2.5 km/sec. Transform to km (oq considers this in km)
        setattr(sctx, 'z2pt5', np.array([z2pt5], dtype='float64') / 1000)

        for key in case.keys():
            if key in site_param_dt and key not in site_keys_defined:
                temp = np.array([case[key]])
                setattr(sctx, key, temp)

        return sctx

    def _set_distances_context(
            self, case: dict, width, dip, ztor) -> DistancesContext:
        """Sets distances context for ground shaking intensity models

        Parameters
        ----------
        case : dict
            Rupture scenario data
        width : float
            Rupture width in [km]
        dip : float
            Dip angle of the fault plane, 0 <= Dip <= 90 deg
        ztor : float
            Depth to top of fault rupture in [km]

        Returns
        -------
        DistancesContext
            Distances context for ground shaking intensity models

        Raises
        ------
        ValueError
            No distance parameter is defined
        """
        dist_keys_defined = {'rrup', 'rx', 'ry0', 'rjb'}

        dctx = contexts.DistancesContext(case)

        # Hanging-wall factor
        if 'fhw' in case.keys():
            fhw = case['fhw']
        else:
            fhw = 0

        # Source-to-site azimuth, alternative of hanging wall factor
        if 'azimuth' in case.keys():
            azimuth = case['azimuth']
        else:
            if fhw == 1:
                azimuth = 50
            elif fhw == 0:
                azimuth = -50

        # rjb and rx
        if 'rjb' in case.keys():
            rjb = case['rjb']
            if rjb == 0:
                rx = 0.5 * width * np.cos(np.radians(dip))
            else:
                if dip == 90:
                    rx = rjb * np.sin(np.radians(azimuth))
                else:
                    if (0 <= azimuth < 90) or (90 < azimuth <= 180):
                        if rjb * np.abs(np.tan(np.radians(azimuth))) \
                                <= width * np.cos(np.radians(dip)):
                            rx = rjb * np.abs(np.tan(np.radians(azimuth)))
                        else:
                            rx = rjb * np.tan(np.radians(azimuth)) \
                                * np.cos(np.radians(azimuth) - np.arcsin(
                                    width * np.cos(np.radians(dip))
                                    * np.cos(np.radians(azimuth)) / rjb))
                    elif azimuth == 90:  # we assume that Rjb>0
                        rx = rjb + width * np.cos(np.radians(dip))
                    else:
                        rx = rjb * np.sin(np.radians(azimuth))
        elif 'rx' in case.keys():
            rx = case['rx']
            rjb = None
        else:
            rx = None

        # ry0
        if azimuth == 90 or azimuth == -90:
            ry0 = 0
        elif azimuth == 0 or azimuth == 180 or azimuth == -180 and rjb:
            ry0 = rjb
        elif rx:
            ry0 = np.abs(rx * 1. / np.tan(np.radians(azimuth)))
        else:
            ry0 = None

        # rrup
        if rjb and dip == 90:
            rrup = np.sqrt(np.square(rjb) + np.square(ztor))
        elif rx:
            if rx < ztor * np.tan(np.radians(dip)):
                rrup1 = np.sqrt(np.square(rx) + np.square(ztor))
            if ztor * np.tan(np.radians(dip)) <= rx <= ztor \
                    * np.tan(np.radians(dip)) + width \
                    * 1. / np.cos(np.radians(dip)):
                rrup1 = rx * np.sin(np.radians(dip)) + \
                    ztor * np.cos(np.radians(dip))
            if rx > ztor * np.tan(np.radians(dip)) \
                    + width * 1. / np.cos(np.radians(dip)):
                rrup1 = np.sqrt(
                    np.square(rx - width * np.cos(np.radians(dip)))
                    + np.square(ztor + width * np.sin(np.radians(dip))))
            rrup = np.sqrt(np.square(rrup1) + np.square(ry0))
        elif 'rrup' not in case.keys():
            if 'rhypo' in case.keys():
                rrup = case['rhypo']
            elif 'repi' in case.keys():
                rrup = case['repi']
            else:
                raise ValueError('No distance parameter is defined!')

        # Closest distance to coseismic rupture (km)
        setattr(dctx, 'rrup', np.array([rrup], dtype='float64'))
        # Horizontal distance from top of rupture measured perpendicular
        # to fault strike (km)
        if rx:
            setattr(dctx, 'rx', np.array([rx], dtype='float64'))
        # The horizontal distance off the end of the rupture measured parallel
        # to strike (km)
        if ry0:
            setattr(dctx, 'ry0', np.array([ry0], dtype='float64'))
        # Closest distance to surface projection of coseismic rupture (km)
        if rjb:
            setattr(dctx, 'rjb', np.array([rjb], dtype='float64'))

        for key in case.keys():
            if key in KNOWN_DISTANCES and key not in dist_keys_defined:
                temp = np.array([case[key]])
                setattr(dctx, key, temp)

        return dctx

    def _set_rupture_context(
            self, case: dict) -> Tuple[RuptureContext, float, float, float]:
        """Set rupture calculation context for ground shaking intensity models

        Parameters
        ----------
        case : dict
            Rupture scenario data

        Returns
        -------
        Tuple[RuptureContext, float, float, float]
            Rupture calculation context for ground shaking intensity models
            Rupture width in [km]
            Dip angle of the fault plane, 0 <= Dip <= 90 deg
            Depth to top of fault rupture in [km]
        """
        rup_keys_defined = {
            'rake', 'dip', 'mag', 'width', 'hypo_depth', 'ztor',
            'occurrence_rate'}

        # Initialize, the contexts for the scenario
        rctx = contexts.RuptureContext(case)

        # RUPTURE PARAMETERS
        # Earthquake magnitude
        mag = case['mag']
        # Fault rake
        rake = case['rake']

        # Hypocentral depth
        if 'hypo_depth' in case.keys():
            hypo_depth = case['hypo_depth']
        else:
            if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                hypo_depth = 5.63 + 0.68 * mag
            else:
                hypo_depth = 11.24 - 0.2 * mag

        # Fault dip
        if 'dip' in case.keys():
            dip = case['dip']
        else:
            if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                dip = 90
            elif rake > 0:
                dip = 40
            else:
                dip = 50

        # Upper and lower seismogenic depths
        if 'upper_sd' in case.keys():
            upper_sd = case['upper_sd']
        else:
            upper_sd = 0
        if 'lower_sd' in case.keys():
            lower_sd = case['lower_sd']
        else:
            lower_sd = 500

        # Rupture width and depth to top of coseismic rupture (km)
        if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            width = 10.0 ** (-0.76 + 0.27 * mag)
        elif rake > 0:
            # thrust/reverse
            width = 10.0 ** (-1.61 + 0.41 * mag)
        else:
            # normal
            width = 10.0 ** (-1.14 + 0.35 * mag)

        source_vertical_width = width * np.sin(np.radians(dip))
        ztor = max(hypo_depth - 0.6 * source_vertical_width, upper_sd)
        if (ztor + source_vertical_width) > lower_sd:
            source_vertical_width = lower_sd - ztor
            width = source_vertical_width / np.sin(np.radians(dip))
        if 'width' in case.keys():
            width = case['width']
        if 'ztor' in case.keys():
            ztor = case['ztor']

        # Fault rake
        setattr(rctx, 'rake', np.array([rake], dtype='float64'))
        # Fault dip
        setattr(rctx, 'dip', np.array([dip], dtype='float64'))
        # Earthquake magnitude
        setattr(rctx, 'mag', np.array([mag], dtype='float64'))
        # Rupture width
        setattr(rctx, 'width', np.array([width], dtype='float64'))
        # Hypocentral depth of the rupture
        setattr(rctx, 'hypo_depth', np.array([hypo_depth], dtype='float64'))
        # Depth to top of coseismic rupture (km)
        setattr(rctx, 'ztor', np.array([ztor], dtype='float64'))
        # Annual rate of occurrence, not really required, setting this to zero
        setattr(rctx, 'occurrence_rate', np.array([0.0], dtype='float64'))

        # Do another loop in case some other rupture parameters
        for key in case.keys():
            if key in RUPTURE_PARAMETERS and key not in rup_keys_defined:
                temp = np.array([case[key]])
                setattr(rctx, key, temp)

        return rctx, width, dip, ztor

    def _set_contexts(
        self, case: dict
    ) -> Tuple[SitesContext, RuptureContext, DistancesContext]:
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
        sctx : .gsim.contexts.SitesContext
            An instance of SitesContext with sites information to calculate
            PoEs on.
        rctx : .gsim.contexts.RuptureContext
            An instance of RuptureContext with a single rupture information.
        dctx : .gsim.contexts.DistancesContext
            An instance of DistancesContext with information about the
            distances between sites and a rupture
        """

        return Context(case)

    def _get_supported_parameters(self, which: str):
        if which == "rupture":
            return RUPTURE_PARAMETERS
        elif which == "sites":
            return site_param_dt
        elif which == "distances":
            return KNOWN_DISTANCES
        else:
            raise ValueError(f"Parameter type {which} not recognized")

    def get_available_gsims(self):
        return inspect_file_for_classes(gsim_models)

    def check_gmpe_attributes(self, gmpe: str):

        model = self._validate_gmm(gmpe)

        for attr, description in self.MODEL_ATTRIBUTES.items():
            if hasattr(model, attr):
                value = getattr(model, attr)

                print(f"{description}: {value}")

    def get_gmpe_attributes(self, gmpe: str):
        model = self._validate_gmm(gmpe)

        parameters = set()
        for attr, description in self.MODEL_ATTRIBUTES.items():
            if attr in ["REQUIRES_SITES_PARAMETERS",
                        "REQUIRES_RUPTURE_PARAMETERS",
                        "REQUIRES_DISTANCES"] and hasattr(model, attr):
                value = getattr(model, attr)

                parameters = parameters | value

        return parameters


def compute_hazard_maps(curves, imls, poes):
    """
    Given a set of hazard curve poes, interpolate hazard maps at the specified
    ``poes``.

    :param curves:
        Array of floats of shape N x L. Each row represents a curve, where the
        values in the row are the PoEs (Probabilities of Exceedance)
        corresponding to the ``imls``.
        Each curve corresponds to a geographical location.
    :param imls:
        Intensity Measure Levels associated with these hazard ``curves``. Type
        should be an array-like of floats.
    :param poes:
        Value(s) on which to interpolate a hazard map from the input
        ``curves``.
    :returns:
        An array of shape N x P, where N is the number of curves and P the
        number of poes.
    """
    # cutoff value for the poe
    EPSILON = 1E-30

    import warnings

    P = len(poes)
    N, L = curves.shape  # number of levels
    if L != len(imls):
        raise ValueError('The curves have %d levels, %d were passed' %
                         (L, len(imls)))

    log_poes = np.log(poes)
    hmap = np.zeros((N, P))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # avoid RuntimeWarning: divide by zero for zero levels
        imls = np.log(np.array(imls[::-1]))
    for n, curve in enumerate(curves):
        # the hazard curve, having replaced the too small poes with EPSILON
        log_curve = np.log([max(poe, EPSILON) for poe in curve[::-1]])
        for p, log_poe in enumerate(log_poes):
            if log_poe > log_curve[-1]:
                # special case when the interpolation poe is bigger than the
                # maximum, i.e the iml must be smaller than the minimum;
                # extrapolate the iml to zero as per
                # https://bugs.launchpad.net/oq-engine/+bug/1292093;
                # then the hmap goes automatically to zero
                pass
            else:
                # exp-log interpolation, to reduce numerical errors
                # see https://bugs.launchpad.net/oq-engine/+bug/1252770
                hmap[n, p] = np.exp(np.interp(log_poe, log_curve, imls))
    return hmap
