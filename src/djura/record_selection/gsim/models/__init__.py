# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

# flake8: noqa

# 1-) ACTIVE SHALLOW CRUSTAL ZONE GMPES
# Pan-European GMPEs: CAV, IA
from .sandikkaya_akkar_2017 import (
    SandikkayaAkkar2017Repi, SandikkayaAkkar2017Rhyp, SandikkayaAkkar2017Rjb)
# Pan-European GMPEs: PGA, PGV, SA
from .derras_2014 import DerrasEtAl2014, DerrasEtAl2014RhypoGermany
from .akkar_bommer_2010 import AkkarBommer2010
from .akkar_2014 import (
    AkkarEtAlRjb2014, AkkarEtAlRhyp2014, AkkarEtAlRepi2014)
# Pan-European GMPEs: PGA, PGV, SA
from .kotha_2020 import KothaEtAl2020, KothaEtAl2020ESHM20
# Pan-European GMPEs: PGA, SA
from .eshm20_craton import ESHM20Craton
# Local GMPEs for Switzerland: PGA, SA
from .akkar_bommer_2010 import (
    AkkarBommer2010SWISS01, AkkarBommer2010SWISS04, AkkarBommer2010SWISS08)
# Local GMPEs for Italy: PGA, PGV, SA
from .bindi_2011 import (
    BindiEtAl2011, BindiEtAl2011Ita19Low, BindiEtAl2011Ita19Upp)
from .lanzano_luzi_2019 import LanzanoLuzi2019shallow, LanzanoLuzi2019deep
# Local GMPEs for Turkiye: PGA, PGV, SA
from .kale_2015 import KaleEtAl2015Turkey
from .akkar_cagnan_2010 import AkkarCagnan2010
# Local GMPEs for Iran: PGA, PGV, SA
from .kale_2015 import KaleEtAl2015Iran
# Local GMPEs for North East US: PGA, PGV, SA
from .atkinson_boore_2006 import (
    AtkinsonBoore2006,
    AtkinsonBoore2006Modified2011,
    AtkinsonBoore2006SGS,
    AtkinsonBoore2006MblgAB1987bar140NSHMP2008,
    AtkinsonBoore2006MblgAB1987bar200NSHMP2008,
    AtkinsonBoore2006MblgJ1996bar140NSHMP2008,
    AtkinsonBoore2006MblgJ1996bar200NSHMP2008,
    AtkinsonBoore2006Mwbar140NSHMP2008,
    AtkinsonBoore2006Mwbar200NSHMP2008)
# Global GMPEs: RSD575, RSD595
from .bommer_2009 import BommerEtAl2009RSD
from .afshari_stewart_2016 import AfshariStewart2016, AfshariStewart2016Japan
from .abrahamson_silva_1996 import AbrahamsonSilva1996
# Global GMPEs from NGA-West 1 Project: PGA, PGV, SA
from .abrahamson_silva_2008 import AbrahamsonSilva2008
from .boore_atkinson_2008 import BooreAtkinson2008
from .boore_atkinson_2011 import BooreAtkinson2011
from .chiou_youngs_2008 import ChiouYoungs2008
# Global GMPEs from NGA-West 1 Project: PGA, PGV, PGD, SA, CAV
from .campbell_bozorgnia_2008 import (
    CampbellBozorgnia2008, CampbellBozorgnia2008Arbitrary)
# Global GMPEs from NGA-West 2 Project: PGA, SA
from .idriss_2014 import Idriss2014
# Global GMPEs from NGA-West 2 Project: PGA, PGV, SA
from .abrahamson_2014 import (
    AbrahamsonEtAl2014,
    AbrahamsonEtAl2014RegCHN,
    AbrahamsonEtAl2014RegJPN,
    AbrahamsonEtAl2014RegTWN)
from .boore_2014 import (
    BooreEtAl2014,
    BooreEtAl2014CaliforniaBasin,
    BooreEtAl2014CaliforniaBasinNoSOF,
    BooreEtAl2014HighQ,
    BooreEtAl2014HighQCaliforniaBasin,
    BooreEtAl2014HighQCaliforniaBasinNoSOF,
    BooreEtAl2014HighQJapanBasin,
    BooreEtAl2014HighQJapanBasinNoSOF,
    BooreEtAl2014HighQNoSOF,
    BooreEtAl2014JapanBasin,
    BooreEtAl2014JapanBasinNoSOF,
    BooreEtAl2014LowQ,
    BooreEtAl2014LowQCaliforniaBasin,
    BooreEtAl2014LowQCaliforniaBasinNoSOF,
    BooreEtAl2014LowQJapanBasin,
    BooreEtAl2014LowQJapanBasinNoSOF,
    BooreEtAl2014LowQNoSOF,
    BooreEtAl2014NoSOF)
from .chiou_youngs_2014 import (
    ChiouYoungs2014,
    ChiouYoungs2014Italy,
    ChiouYoungs2014Japan,
    ChiouYoungs2014NearFaultEffect,
    ChiouYoungs2014PEER,
    ChiouYoungs2014Wenchuan)
from .campbell_bozorgnia_2014 import (
    CampbellBozorgnia2014,
    CampbellBozorgnia2014HighQ,
    CampbellBozorgnia2014HighQJapanSite,
    CampbellBozorgnia2014JapanSite,
    CampbellBozorgnia2014LowQ,
    CampbellBozorgnia2014LowQJapanSite)
from .boore_2020 import BooreEtAl2020
from .bozorgnia_campbell_2016 import BozorgniaCampbell2016
# Global GMPE: PGA, PGV, PGD, SA, Sa_avg2, Sa_avg3, FIV3, Ds575, Ds595
from .aristeidou_2024 import AristeidouEtAl2024

# Global GMPE: FIV3
from .davalos_2020 import DavalosEtAl2020

# Global GMPE: IA
from .bahrampouri_2021 import (
    BahrampouriEtAl2021Asc,
    BahrampouriEtAl2021SInter,
    BahrampouriEtAl2021SSlab)
from .travasarou_2003 import TravasarouEtAl2003

# 2-) SUBDUCTION ZONE GMPES
# Global GMPEs: PGA, SA
from .abrahamson_gulerce_2020 import (
    AbrahamsonGulerce2020SInter,
    AbrahamsonGulerce2020SInterAlaska,
    AbrahamsonGulerce2020SInterCascadia,
    AbrahamsonGulerce2020SInterCentralAmericaMexico,
    AbrahamsonGulerce2020SInterJapan,
    AbrahamsonGulerce2020SInterNewZealand,
    AbrahamsonGulerce2020SInterSouthAmerica,
    AbrahamsonGulerce2020SInterTaiwan,
    AbrahamsonGulerce2020SSlab,
    AbrahamsonGulerce2020SSlabAlaska,
    AbrahamsonGulerce2020SSlabCascadia,
    AbrahamsonGulerce2020SSlabCentralAmericaMexico,
    AbrahamsonGulerce2020SSlabJapan,
    AbrahamsonGulerce2020SSlabNewZealand,
    AbrahamsonGulerce2020SSlabSouthAmerica,
    AbrahamsonGulerce2020SSlabTaiwan)
from .abrahamson_2015 import AbrahamsonEtAl2015SInter, AbrahamsonEtAl2015SSlab
# Local GMPEs for New Zealand: RSD575
from .vanhoutte_2018 import VanHoutteEtAl2018RSD
# Pan-European GMPEs: PGA, SA
from .bchydro_2016_epistemic import BCHydroESHM20SInter, BCHydroESHM20SSlab
from .cauzzi_faccioli_2008 import CauzziFaccioli2008, FaccioliEtAl2010

# Indirect AVGSA - Generic GMPE
from .gmpe_avgsa import GmpeIndirectAvgSA
