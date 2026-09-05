# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

from enum import Enum
import numpy as np


class TRT(Enum):
    """
    Container for constants that define some of the common Tectonic Region
    Types.
    """
    # Constant values correspond to the NRML schema definition.
    ACTIVE_SHALLOW_CRUST = 'Active Shallow Crust'
    STABLE_CONTINENTAL = 'Stable Shallow Crust'
    SUBDUCTION_INTERFACE = 'Subduction Interface'
    SUBDUCTION_INTRASLAB = 'Subduction IntraSlab'
    UPPER_MANTLE = "Upper Mantle"
    VOLCANIC = 'Volcanic'
    GEOTHERMAL = 'Geothermal'
    INDUCED = 'Induced'


# NB: cannot be an enum because it would break the Strong Motion Toolkit :-(
class StdDev(object):
    """
    GSIM standard deviation represents ground shaking variability at a site.
    """
    TOTAL = 'Total'
    #: Standard deviation representing ground shaking variability
    #: within different events.
    INTER_EVENT = 'Inter event'
    #: Standard deviation representing ground shaking variability
    #: within a single event.
    INTRA_EVENT = 'Intra event'
    #: Total standard deviation, defined as the square root of the sum
    #: of inter- and intra-event squared standard deviations, represents
    #: the total ground shaking variability, and is the only one that
    #: is used for calculating a probability of intensity exceedance
    EVENT = 'Event'
    #: Used in event based calculations, correspond to TOTAL if the gsim
    #: is defined for TOTAL, otherwise to the pair (INTER_EVENT, INTRA_EVENT)
    ALL = 'All'
    #: Compute all the standard deviations for which the GMPE is defined


class IMC(Enum):
    """
    The intensity measure component is the component of interest
    of ground shaking
    """
    #: The horizontal component.
    HORIZONTAL = 'Horizontal'
    #: The median horizontal component.
    MEDIAN_HORIZONTAL = 'Median horizontal'
    #: Usually defined as the geometric average of the maximum
    #: of the two horizontal components (which may not occur
    #: at the same time).
    GEOMETRIC_MEAN = 'Average Horizontal'
    #: An orientation-independent alternative to :attr:`AVERAGE_HORIZONTAL`.
    #: Defined at Boore et al. (2006, Bull. Seism. Soc. Am. 96, 1502-1511)
    #: and is used for all the NGA GMPEs.
    GMRotI50 = 'Average Horizontal (GMRotI50)'
    #: The geometric mean of the records rotated into the most adverse
    #: direction for the structure.
    GMRotD100 = "Average Horizontal (GMRotD100)"
    #: An orientation-independent alternative to :attr:`AVERAGE_HORIZONTAL`.
    #: Defined at Boore et al. (2006, Bull. Seism. Soc. Am. 96, 1502-1511)
    #: and is used for all the NGA GMPEs.
    RotD50 = 'Average Horizontal (RotD50)'
    #:
    RotD100 = 'Horizontal Maximum Direction (RotD100)'
    #: A randomly chosen horizontal component.
    RANDOM_HORIZONTAL = 'Random horizontal'
    #: The largest value obtained from two perpendicular horizontal
    #: components.
    GREATER_OF_TWO_HORIZONTAL = 'Greater of two horizontal'
    #: The vertical component.
    VERTICAL = 'Vertical'
    #: "Vectorial addition: a_V = sqrt(max|a_1(t)|^2 + max|a_2(t)|^2)).
    #: This means that the maximum ground amplitudes occur simultaneously on
    #: the two horizontal components; this is a conservative assumption."
    #: p. 53 of Douglas (2003, Earth-Sci. Rev. 61, 43-104)
    VECTORIAL = 'Square root of sum of squares of peak horizontals'
    #: "the peak square root of the sum of squares of two orthogonal
    #: horizontal components in the time domain"
    #: p. 880 of Kanno et al. (2006, Bull. Seism. Soc. Am. 96, 879-897)
    PEAK_SRSS_HORIZONTAL = 'Peak square root of sum of squares of horizontals'
    #: A vertical-to-horizontal spectral ratio
    VERTICAL_TO_HORIZONTAL_RATIO = 'Vertical-to-Horizontal Ratio'


ampcode_dt = (np.bytes_, 4)

site_param_dt = {
    'sids': np.uint32,
    'site_id': np.uint32,
    'lon': np.float64,
    'lat': np.float64,
    'depth': np.float64,
    'vs30': np.float64,
    'kappa0': np.float64,
    'vs30measured': bool,
    'z1pt0': np.float64,
    'z2pt5': np.float64,
    'siteclass': (np.bytes_, 1),
    'geohash': (np.bytes_, 6),
    'z1pt4': np.float64,
    'backarc': np.uint8,  # 0=forearc,1=backarc,2=alongarc
    'xvf': np.float64,
    'soiltype': np.uint32,
    'bas': bool,

    # Parameters for site amplification
    'ampcode': ampcode_dt,
    'ec8': (np.bytes_, 1),
    'ec8_p18': (np.bytes_, 2),
    'h800': np.float64,
    'geology': (np.bytes_, 20),
    'amplfactor': np.float64,
    'fpeak': np.float64,
    # Fundamental period and and amplitude of HVRSR spectra
    'THV': np.float64,
    'PHV': np.float64,

    # parameters for secondary perils
    'friction_mid': np.float64,
    'cohesion_mid': np.float64,
    'saturation': np.float64,
    'dry_density': np.float64,
    'Fs': np.float64,
    'crit_accel': np.float64,
    'unit': (np.bytes_, 5),
    'liq_susc_cat': (np.bytes_, 2),
    'dw': np.float64,
    'yield_acceleration': np.float64,
    'slope': np.float64,
    'gwd': np.float64,
    'cti': np.float64,
    'dc': np.float64,
    'dr': np.float64,
    'dwb': np.float64,
    'hwater': np.float64,
    'precip': np.float64,

    # parameters for YoudEtAl2002
    'freeface_ratio': np.float64,
    'T_15': np.float64,
    'D50_15': np.float64,
    'F_15': np.float64,
    'T_eq': np.float64,

    # other parameters
    'custom_site_id': (np.bytes_, 8),
    'region': np.uint32,
    'in_cshm': bool  # used in mcverry
}

KNOWN_DISTANCES = frozenset(
    'rrup rx ry0 rjb rhypo repi rcdpp azimuth azimuth_cp rvolc closest_point'
    .split())

RUPTURE_PARAMETERS = frozenset({
    "mag", "strike", "dip", "rake", "ztor", "hypo_lon", "hypo_lat",
    "hypo_depth", "width", "in_cshm", "zbot", "occurrence_rate",
})
