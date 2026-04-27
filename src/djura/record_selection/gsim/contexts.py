# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import abc
import numpy as np
import copy as cp


MECHANISM_MAP = {
    0: 'strike-slip fault',
    1: 'normal fault',
    2: 'reverse fault',
    3: 'reverse/oblique fault',
    4: 'normal/oblique fault'
}

MECHANISM_MAP_REV = {
    'strike-slip fault': 0,
    'normal fault': 1,
    'reverse fault': 2,
    'reverse/oblique fault': 3,
    'normal/oblique fault': 4,
}


class BaseContext(metaclass=abc.ABCMeta):
    """
    Base class for context objects.
    """

    _slots_ = ()

    def __init__(self, param_dict: dict):
        for key, value in param_dict.items():
            if key in self._slots_:
                setattr(self, key, np.atleast_1d(value))

        # Map the fault mechanism to integers
        if hasattr(self, "mechanism"):
            self.mechanism = np.vectorize(MECHANISM_MAP_REV.get)(
                self.mechanism)

    def __eq__(self, other):
        """
        Returns
        -------
        bool
            True if ``other`` has same attributes with same values.
        """
        if isinstance(other, self.__class__):
            if self._slots_ == other._slots_:
                oks = []
                for s in self._slots_:
                    a, b = getattr(self, s, None), getattr(other, s, None)
                    if a is None and b is None:
                        ok = True
                    elif a is None and b is not None:
                        ok = False
                    elif a is not None and b is None:
                        ok = False
                    elif hasattr(a, "shape") and hasattr(b, "shape"):
                        if a.shape == b.shape:
                            ok = np.allclose(a, b)
                        else:
                            ok = False
                    else:
                        ok = a == b
                    oks.append(ok)
                return np.all(oks)
        return False

    def copy(self, deep=False):
        """
        Create a copy of the current object.

        Parameters
        ----------
        deep : bool, optional
            If True, performs a deep copy. Otherwise, performs a shallow copy.

        Returns
        -------
        MyClass
            A shallow or deep copy of the current object.
        """
        if deep:
            return cp.deepcopy(self)
        else:
            return cp.copy(self)


class RuptureContext(BaseContext):
    """
    Rupture context for ground motion prediction equations.

    Instances of this class are passed into:`GMPE.get_mean_and_stddevs`.
    They are intended to represent relevant features a single rupture.
    Every GMPE class is required to declare what rupture parameters
    `GMPE.RUPTURE_PARAMETERS` does it need.
    """

    _slots_ = (
        "mag",
        "strike",
        "dip",
        "rake",
        "ztor",
        "hypo_lon",
        "hypo_lat",
        "hypo_depth",
        "width",
        "hypo_loc",
        "d_hyp",
        "in_cshm",
        "mechanism",
        "upper_sd",
        "lower_sd"
    )
    """Available slots for the RuptureContext."""
    mag: float | np.ndarray
    """Magnitude of the rupture."""
    ztor: float | np.ndarray
    """Depth of rupture's top edge (km)."""
    hypo_depth: float | np.ndarray
    """Hypocentral depth from the earthquake (km)."""
    d_hyp: float | np.ndarray
    """Hypocentral depth from the earthquake (km)."""
    rake: float | np.ndarray
    """Angle describing the slip propagation on the rupture surface (degrees).
    """
    dip: float | np.ndarray
    """Rupture's surface dip angle (degrees)."""
    width: float | np.ndarray
    """Down-dip rupture width (km)."""
    hypo_lon: float | np.ndarray
    """Hypocentre longitude (degrees)."""
    hypo_lat: float | np.ndarray
    """Hypocentre latitude (degrees)."""
    in_cshm: int | np.ndarray
    """Parameter to implement modifications required for the Canterbury
    Seismic Hazard Model."""
    mechanism: str | np.ndarray
    """Faulting mechanism (e.g., strike-slip fault)."""
    upper_sd: float | np.ndarray
    """Upper seismogenic depth."""
    lower_sd: float | np.ndarray
    """Lower seismogenic depth."""

    def _set_default_rups(self) -> None:
        """
        Set default rupture parameters with proper handling for scalar or
        array values.
        """

        if not hasattr(self, "rake") and hasattr(self, "mechanism"):
            # Fault rake
            self.rake = np.where(
                self.mechanism == 0,
                0.0,
                np.where(
                    (self.mechanism == 2) | (self.mechanism == 3),
                    90,
                    -90
                )
            )

        # Upper and lower seismogenic depths
        if not hasattr(self, "upper_sd") and hasattr(self, "mag"):
            self.upper_sd = np.zeros_like(self.mag)
        if not hasattr(self, "lower_sd") and hasattr(self, "mag"):
            self.lower_sd = np.full_like(self.mag, 500)

        if hasattr(self, "rake"):
            # 1st mask for identifying strike-slip faulting
            mask1 = (
                ((-45 <= self.rake) and (self.rake <= 45))
                or (self.rake >= 135) or (self.rake <= -135)
            )
            # 2nd mask for identifying Thrust/Reverse faulting
            mask2 = self.rake > 0

            # Fault dip
            if not hasattr(self, "dip"):
                self.dip = np.where(
                    mask1,
                    90,  # Strike-slip
                    np.where(
                        mask2,
                        40,  # Thrust/Reverse
                        50  # Normal
                    ),
                )

            # Hypocentral depth
            if not hasattr(self, "hypo_depth") and hasattr(self, "mag"):
                self.hypo_depth = np.where(
                    mask1,
                    5.63 + 0.68 * self.mag,  # Strike-slip
                    11.24 - 0.2 * self.mag,  # Thrust/Reverse/Normal
                )

            # Rupture width and depth to top of coseismic rupture (km)
            if not hasattr(self, "width") and hasattr(self, "mag"):
                self.width = np.where(
                    mask1,
                    10.0 ** (-0.76 + 0.27 * self.mag),  # Strike-slip
                    np.where(
                        mask2,
                        10.0 ** (-1.61 + 0.41 * self.mag),  # Thrust/Reverse
                        10.0 ** (-1.14 + 0.35 * self.mag)   # Normal
                    ),
                )

        # Depth of rupture's top edge
        if not hasattr(self, "ztor") and hasattr(self, "hypo_depth") \
           and hasattr(self, "upper_sd") and hasattr(self, "width") \
           and hasattr(self, "dip"):

            source_vertical_width = self.width * np.sin(np.radians(self.dip))
            self.ztor = np.maximum(
                self.hypo_depth - 0.6 * source_vertical_width,
                self.upper_sd
            )

        # Adjust rupture width based on ztor and lower_sd
        if hasattr(self, "ztor") and hasattr(self, "lower_sd") \
           and hasattr(self, "width") and hasattr(self, "dip"):

            source_vertical_width = self.width * np.sin(np.radians(self.dip))
            self.width = np.where(
                self.ztor + source_vertical_width > self.lower_sd,
                (self.lower_sd - self.ztor) / np.sin(np.radians(self.dip)),
                self.width
            )


class SitesContext(BaseContext):
    """
    Sites context for ground motion prediction equations.

    Instances of this class are passed into:`GMPE.get_mean_and_stddevs`.
    They are intended to represent relevant features of the site.
    Every GMPE class is required to declare what sites parameters
    `GMPE.SITES_PARAMETERS` does it need.
    """

    _slots_ = (
        "vs30",
        "vs30measured",
        "z1pt0",
        "z1pt4",
        "z2pt5",
        "region",
        "xvf",
        "soiltype",
        "geology",
        "siteclass",
        "h800",
        "ec8_p18",
        "ec8",
        "kappa0",
        "lat",
        "lon",
        "THV",
        "PHV",
        "bas",
        "fpeak",
        "T_15",
        "F_15",
        "D50_15",
        "freeface_ratio",
        "soil",
        "backarc"
    )
    """Available slots for the SitesContext."""
    vs30: float | np.ndarray
    """The average shear-wave velocity (m/s) over a subsurface depth of 30 m.
    """
    vs30measured: int | bool | np.ndarray
    """Flag indicating whether Vs30 is measured (1) or inferred (0)."""
    z1pt0: float | np.ndarray
    """Depth (m) to Vs=1.0 km/sec."""
    z1pt4: float | np.ndarray
    """Depth (m) to Vs=1.4 km/sec."""
    z2pt5: float | np.ndarray
    """Depth (km) to Vs=2.5 km/sec."""
    region: int | np.ndarray
    """Integer value between 0 and 5 indicating the residual attenuation
    region to which the site belongs:
    1: Central/Slower, 2: Central/Fast, 3: Fast, 4: Central, 5: Very Slow,
    0: Default."""
    xvf: float | np.ndarray
    """Distance to the volcanic front (km, positive in the forearc)."""
    soiltype: int | np.ndarray
    """Site class definition (1-6, 1=rock) according to IdiniEtAl2017."""
    slope: float | np.ndarray
    """The topographic slope (%)."""
    geology: str | np.ndarray
    """Local geology definition according to ESHM2020
    (e.g., UNKNOWN, PRECAMBRIAN, PALEOZOIC, etc.)."""
    siteclass: str | np.ndarray
    """Site class definition according to Eurocode 8 or New Zealand code."""
    h800: float | np.ndarray
    """The depth (m) to the seismic bedrock formation, where Vs is at least
    800 m/s."""
    f0: float | np.ndarray
    """The fundamental frequency (Hz) of resonance (if unknown set as 15 for
    rock sites)."""
    ec8_p18: str | np.ndarray
    """The site class definition according to PitilakisEtAl2018."""
    ec8: str | np.ndarray
    """Site Class definition according to CEN2024, indicating that Vs30 and
    h800 are not known but a Eurocode 8 site class is."""
    backarc: int | np.ndarray
    """Flag indicating whether a site is on the forearc (0) or backarc (1)."""
    kappa0: float | np.ndarray
    """Near-surface attenuation parameter."""
    lat: float | np.ndarray
    """Site latitude."""
    lon: float | np.ndarray
    """Site longitude."""
    THV: float | np.ndarray
    """Predominant site period."""
    PHV: float | np.ndarray
    """Amplitude of the peak of the mean HVRSR (horizontal-to-vertical
    response ratios of 5%-damped response spectra)."""
    bas: int | np.ndarray
    """Parameter for applying basin effect correction (0: None, 1: Po-Plain).
    """
    fpeak: float | np.ndarray
    """Peak frequency of the horizontal-to-vertical spectral ratio."""
    soil: int | np.ndarray
    """Soil/rock indicator (1 = soil, 0 = rock)."""

    def _set_defaults_sites(self) -> None:

        if not hasattr(self, "vs30measured") and hasattr(self, "vs30"):
            self.vs30measured = np.full(len(self.vs30), True, dtype=bool)

        if not hasattr(self, "z1pt0") and hasattr(self, "vs30"):
            # in meters (m)

            # Chiou and Youngs 2014, eq. (1) - California and Non-Japan
            self.z1pt0 = np.exp(
                -7.15 / 4
                * np.log((self.vs30**4 + 571**4) / (1360**4 + 571**4))
            )

            # Chiou and Youngs 2014, eq. (2) - Japan
            # self.z1pt0 = np.exp(
            #     -5.23 / 2
            #     * np.log((self.vs30**2 + 412**2) / (1360**2 + 412**2)))

        if not hasattr(self, "z2pt5") and hasattr(self, "vs30"):
            # in kilometers (km)

            # Campbell and Bozorgnia 2014, eq. (33) - California and Non-Japan
            self.z2pt5 = np.exp(7.089 - 1.144 * np.log(self.vs30))

            # Campbell and Bozorgnia 2014, eq. (33) - Japan
            # self.z2pt5 = np.exp(5.359 - 1.102 * np.log(self.vs30))


class DistancesContext(BaseContext):
    """
    Distances context for ground motion prediction equations.

    Instances of this class are passed into:`GMPE.get_mean_and_stddevs`.
    They are intended to represent relevant distances between the site the
    rupture. Every GMPE class is required to declare what distance measures
    `GMPE.REQUIRES_DISTANCES` does it need.
    """

    _slots_ = (
        "rrup",
        "rx",
        "rjb",
        "rhypo",
        "repi",
        "ry0",
        "rcdpp",
        "azimuth",
        "hanging_wall",
        "rvolc",
    )
    """Available slots for the DistancesContext."""
    rhypo: float | np.ndarray
    """Closest distance to the earthquake hypocenter (km)."""
    repi: float | np.ndarray
    """Closest distance on the ground surface between the site and the
    earthquake epicenter (km)."""
    rrup: float | np.ndarray
    """Closest distance to rupture surface (km)."""
    rjb: float | np.ndarray
    """Closest distance to rupture's surface projection, also known as
    Joyner-Boore distance (km)."""
    ry0: float | np.ndarray
    """Horizontal distance off the end of the rupture measured parallel to
    strike (km)."""
    rx: float | np.ndarray
    """Perpendicular distance to rupture top edge projection (km)."""
    azimuth: float | np.ndarray
    """Source-to-station azimuth measured from the strike of the fault plane
    clockwise (degrees)."""
    rvolc: float | np.ndarray
    """Source to site distance passing through surface projection of volcanic
    zone (km)."""
    rcdpp: float | np.ndarray
    """Direct point parameter for directivity effect centered on the site- and
    earthquake-specific average DPP used."""

    def _set_default_dists(
        self,
        width: np.ndarray,
        dip: np.ndarray,
        ztor: np.ndarray
    ) -> None:
        # Hanging-wall factor (default is 0)
        if not hasattr(self, 'fhw'):
            self.fhw = np.zeros_like(width)

        # Azimuth based on hanging-wall factor
        if not hasattr(self, 'azimuth') and hasattr(self, 'fhw'):
            self.azimuth = np.where(self.fhw == 0, -50, 50)

        # rx calculation
        if not hasattr(self, 'rx') and hasattr(self, 'rjb') \
           and hasattr(self, 'azimuth'):

            rjb = np.asarray(self.rjb)
            mask1 = rjb == 0
            mask2 = dip == 90
            mask3 = (0 <= self.azimuth < 90) or (90 < self.azimuth <= 180)
            mask4 = (rjb * np.abs(np.tan(np.radians(self.azimuth)))
                     <= width * np.cos(np.radians(dip)))
            mask5 = self.azimuth == 90  # we assume that Rjb>0
            self.rx = np.where(
                mask1,
                0.5 * width * np.cos(np.radians(dip)),
                np.where(
                    mask2,
                    rjb * np.sin(np.radians(self.azimuth)),
                    np.where(
                        mask3,
                        np.where(
                            mask4,
                            rjb * np.abs(np.tan(np.radians(self.azimuth))),
                            rjb * np.tan(np.radians(self.azimuth))
                            * np.cos(
                                np.radians(self.azimuth)
                                - np.arcsin(
                                    width
                                    * np.cos(np.radians(dip))
                                    * np.cos(np.radians(self.azimuth))
                                    / rjb))
                        ),
                        np.where(
                            mask5,
                            rjb + width * np.cos(np.radians(dip)),
                            rjb * np.sin(np.radians(self.azimuth))
                        )
                    )
                )
            )

        # ry0 calculation
        if not hasattr(self, 'ry0'):
            mask1 = self.azimuth == 90 or self.azimuth == -90
            mask2 = (
                self.azimuth == 0
                or self.azimuth == 180
                or self.azimuth == -180
            ) and np.full(len(self.azimuth), hasattr(self, "rjb"), dtype=bool)
            mask3 = np.full(len(self.azimuth), hasattr(self, "rx"), dtype=bool)

            if np.all(mask1 or mask2 or mask3):
                # Init rjb and rx arrays to avoid issues
                if hasattr(self, "rjb"):
                    rjb = self.rjb
                else:
                    rjb = np.zeros_like(ztor)
                if hasattr(self, "rx"):
                    rx = self.rx
                else:
                    rx = np.zeros_like(ztor)

                self.ry0 = np.where(
                    mask1,
                    0.0,
                    np.where(
                        mask2,
                        rjb,
                        np.abs(rx * 1. / np.tan(np.radians(self.azimuth)))
                    )
                )

        # rrup calculation
        if not hasattr(self, 'rrup'):
            mask1 = (np.full(len(dip), hasattr(self, "rjb"), dtype=bool)
                     and dip == 90)
            mask2 = np.full(len(dip), hasattr(self, "rx"), dtype=bool)
            mask3 = np.full(len(dip), hasattr(self, "rhypo"), dtype=bool)
            mask4 = np.full(len(dip), hasattr(self, "repi"), dtype=bool)
            if np.all(mask1 or mask2 or mask3 or mask4):
                # Init rjb array to avoid issues
                if hasattr(self, "rjb"):
                    rjb = self.rjb
                else:
                    rjb = np.zeros_like(ztor)

                self.rrup = np.where(
                    mask1,
                    np.sqrt(np.square(rjb) + np.square(ztor)),
                    np.nan
                )

                if np.any(np.isnan(self.rrup)) and hasattr(self, "rx"):
                    rrup1 = np.ones_like(ztor)
                    mask5 = self.rx < ztor * np.tan(np.radians(dip))
                    mask6 = ztor * np.tan(np.radians(dip)) <= self.rx <= ztor \
                        * np.tan(np.radians(dip)) + width \
                        * 1. / np.cos(np.radians(dip))
                    mask7 = self.rx > ztor * np.tan(np.radians(dip)) \
                        + width * 1. / np.cos(np.radians(dip))

                    rrup1[mask5] = np.sqrt(
                        np.square(self.rx) + np.square(ztor))
                    rrup1[mask6] = self.rx * np.sin(np.radians(dip)) + \
                        ztor * np.cos(np.radians(dip))
                    rrup1[mask7] = np.sqrt(
                        np.square(self.rx - width * np.cos(np.radians(dip)))
                        + np.square(ztor + width * np.sin(np.radians(dip))))

                    self.rrup = np.where(
                        np.isnan(self.rrup),
                        np.sqrt(np.square(rrup1) + np.square(self.ry0)),
                        self.rrup)

                if np.any(np.isnan(self.rrup)) and hasattr(self, "rhypo"):
                    self.rrup = np.where(
                        np.isnan(self.rrup),
                        self.rhypo,
                        self.rrup)

                if np.any(np.isnan(self.rrup)) and hasattr(self, "repi"):
                    self.rrup = np.where(
                        np.isnan(self.rrup),
                        self.repi,
                        self.rrup)


class Context(RuptureContext, DistancesContext, SitesContext):
    """
    A comprehensive context class that combines the rupture, site, and distance
    contexts for ground motion prediction equations (GMPEs).

    This class inherits from `RuptureContext`, `DistancesContext`, and
    `SitesContext`, combining all relevant parameters related to earthquake
    rupture, site conditions, and distances between the seismic source and the
    site.
    """

    _slots_ = RuptureContext._slots_ + DistancesContext._slots_ + \
        SitesContext._slots_

    def __init__(self, param_dict: dict):
        """
        Initialize the combined context with given parameters.
        """
        super().__init__(param_dict)
        self._set_default_rups()
        self._set_defaults_sites()
        if hasattr(self, 'width') and hasattr(self, 'dip') \
           and hasattr(self, 'ztor'):
            self._set_default_dists(self.width, self.dip, self.ztor)
        self.ensure_astypes()

    def ensure_astypes(self):
        """
        Ensures that numpy array types are correct.

        Notes
        -----
        OQ mostly set ctx to float16, this can create minor differences
        in the last digits. I think we do not need to be that precise.
        """
        for key, value in self.__dict__.items():
            # vs30measured should be bool
            if key == 'vs30measured':
                self.vs30measured = self.vs30measured.astype(bool)
            # flags should be integer
            elif key == 'mechanism':
                self.mechanism = self.mechanism.astype(np.int8)
            elif key == 'bas':
                self.bas = self.bas.astype(np.int8)
            elif key == 'soil':
                self.soil = self.soil.astype(np.int8)
            elif key == 'soiltype':
                self.soiltype = self.soiltype.astype(np.int8)
            elif key == 'in_cshm':
                self.in_cshm = self.in_cshm.astype(np.int8)
            elif key == 'region':
                self.region = self.region.astype(np.int8)
            elif key == 'backarc':
                self.backarc = self.backarc.astype(np.int8)
            elif isinstance(value, np.ndarray) and (
                np.issubdtype(value.dtype, np.integer)
                or np.issubdtype(value.dtype, np.floating)
            ):
                setattr(self, key, np.array(value, dtype=np.float32))
