# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

from .boore_atkinson_2008 import BooreAtkinson2008
from .. import const


class BooreAtkinson2011(BooreAtkinson2008):
    """
    Implements GMPE based on the corrections proposed by Gail M. Atkinson
    and D. Boore in 2011 and published as "Modifications to Existing
    Ground-Motion Prediction Equations in Light of New Data " (2011,
    Bulletin of the Seismological Society of America, Volume 101, No. 3,
    pages 1121-1135).
    """
    kind = '2011'


class Atkinson2008prime(BooreAtkinson2008):
    """
    Implements the Boore & Atkinson (2011) adjustment to the Atkinson (2008)
    GMPE (not itself implemented in OpenQuake)
    """
    # GMPE is defined for application to Eastern North America (Stable Crust)
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL
    kind = 'prime'
