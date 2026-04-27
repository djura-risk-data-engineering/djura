from typing import Optional, Dict
from pydantic import BaseModel, Field


class KothaEtAl2020Model(BaseModel):
    sigma_mu_epsilon: float = Field(default=0.0)
    c3_epsilon: float = Field(default=0.0)
    ergodic: bool = Field(default=True)
    dl2l: Optional[Dict] = Field(default=None)
    c3: Optional[Dict] = Field(default=None)


class KothaEtAl2020ESHM20Model(KothaEtAl2020Model):
    pass


class AkkarEtAlRjb2014Model(BaseModel):
    adjustment_factor: float = Field(default=1.0)


class DerrasEtAl2014RhypoGermanyModel(AkkarEtAlRjb2014Model):
    pass


class AkkarEtAlRhyp2014Model(AkkarEtAlRjb2014Model):
    pass


class AAkkarEtAlRepi2014Model(AkkarEtAlRjb2014Model):
    pass


class ESHM20CratonModel(BaseModel):
    epsilon: float = Field(default=0.0)
    tau_model: str = Field(default='global')
    phi_model: str = Field(default='global')
    ergodic: bool = Field(default=True)
    tau_quantile: float = Field(default=None)
    phi_ss_quantile: float = Field(default=None)
    site_epsilon: float = Field(default=0.0)


class AtkinsonBoore2006Model(BaseModel):
    mag_eq: str = Field(default='NA')
    scale_fac: float = Field(default=0)


class AtkinsonBoore2006Modified2011Model(AtkinsonBoore2006Model):
    pass


class AtkinsonBoore2006SGSModel(AtkinsonBoore2006Model):
    pass


class AbrahamsonEtAl2014Model(BaseModel):
    sigma_mu_epsilon: float = Field(default=0.0)
    region: str = Field(default=None)


class AbrahamsonEtAl2014RegCHNModel(BaseModel):
    sigma_mu_epsilon: float = Field(default=0.0)


class AbrahamsonEtAl2014RegJPNModel(AbrahamsonEtAl2014RegCHNModel):
    pass


class AbrahamsonEtAl2014RegTWNModel(AbrahamsonEtAl2014RegCHNModel):
    pass


class BooreEtAl2014Model(BaseModel):
    region: str = Field(default='nobasin')
    sof: bool = Field(default=True)
    sigma_mu_epsilon: float = Field(default=0.0)


class BooreEtAl2014CaliforniaBasinModel(BaseModel):
    sigma_mu_epsilon: float = Field(default=0.0)


class BooreEtAl2014LowQCaliforniaBasinModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014LowQCaliforniaBasinNoSOFModel(
        BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014LowQJapanBasinModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014LowQJapanBasinNoSOFModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014LowQNoSOFModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014HighQCaliforniaBasinModel(
        BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014HighQCaliforniaBasinNoSOFModel(
        BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014HighQJapanBasinModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014HighQJapanBasinNoSOFModel(
        BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014HighQNoSOFModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014CaliforniaBasinNoSOFModel(
        BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014JapanBasinModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014JapanBasinNoSOFModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class BooreEtAl2014NoSOFModel(BooreEtAl2014CaliforniaBasinModel):
    pass


class ChiouYoungs2014Model(BaseModel):
    sigma_mu_epsilon: float = Field(default=0.0)
    use_hw: bool = Field(default=True)
    add_delta_c1: bool = Field(default=False)
    alpha_nm: float = Field(default=1.0)
    stress_par_host: float = Field(default=None)
    stress_par_target: float = Field(default=None)


class ChiouYoungs2014ItalyModel(ChiouYoungs2014Model):
    pass


class ChiouYoungs2014JapanModel(ChiouYoungs2014Model):
    pass


class ChiouYoungs2014NearFaultEffectModel(ChiouYoungs2014Model):
    pass


class ChiouYoungs2014PEERModel(ChiouYoungs2014Model):
    pass


class ChiouYoungs2014WenchuanModel(ChiouYoungs2014Model):
    pass


class CampbellBozorgnia2014Model(BaseModel):
    estimate_ztor: bool = Field(default=False)
    estimate_width: bool = Field(default=False)
    estimate_hypo_depth: bool = Field(default=False)


class CampbellBozorgnia2014HighQModel(CampbellBozorgnia2014Model):
    pass


class CampbellBozorgnia2014HighQJapanSiteModel(CampbellBozorgnia2014Model):
    pass


class CampbellBozorgnia2014JapanSiteModel(CampbellBozorgnia2014Model):
    pass


class CampbellBozorgnia2014LowQModel(CampbellBozorgnia2014Model):
    pass


class CampbellBozorgnia2014LowQJapanSiteModel(CampbellBozorgnia2014Model):
    pass


class BozorgniaCampbell2016Model(BaseModel):
    SJ: bool = Field(default=False)
    sgn: int = Field(default=0)


class AbrahamsonGulerce2020SInterModel(BaseModel):
    region: str = Field(default="GLO")
    ergodic: bool = Field(default=True)
    apply_usa_adjustment: bool = Field(default=False)
    sigma_mu_epsilon: float = Field(default=0.0)


class AbrahamsonGulerce2020SInterAlaskaModel(BaseModel):
    ergodic: bool = Field(default=True)
    apply_usa_adjustment: bool = Field(default=False)
    sigma_mu_epsilon: float = Field(default=0.0)


class AbrahamsonGulerce2020SInterCascadiaModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SInterCentralAmericaMexicoModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SInterJapanModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SInterNewZealandModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SInterSouthAmericaModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SInterTaiwanModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabModel(AbrahamsonGulerce2020SInterModel):
    pass


class AbrahamsonGulerce2020SSlabAlaskaModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabCascadiaModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabCentralAmericaMexicoModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabJapanModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabNewZealandModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabSouthAmericaModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonGulerce2020SSlabTaiwanModel(
        AbrahamsonGulerce2020SInterAlaskaModel):
    pass


class AbrahamsonEtAl2015SInterModel(BaseModel):
    ergodic: bool = Field(default=True)
    theta6_adjustment: float = Field(default=0.0)
    sigma_mu_epsilon: float = Field(default=0.0)
    faba_taper_model: str = Field(default="Step")


class AbrahamsonEtAl2015SSlabModel(AbrahamsonEtAl2015SInterModel):
    pass


class BCHydroESHM20SInterModel(BaseModel):
    ergodic: bool = Field(default=True)
    theta6_adjustment: float = Field(default=0.0)
    sigma_mu_epsilon: float = Field(default=0.0)
    faba_taper_model: str = Field(default="SFunc")
    a: float = Field(default=-100.0)
    b: float = Field(default=100.0)


class BCHydroESHM20SSlabModel(BCHydroESHM20SInterModel):
    pass
