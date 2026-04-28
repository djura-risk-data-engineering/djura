import numpy as np
from pydantic import field_validator, ConfigDict, BaseModel, Field, \
    RootModel
from typing import Optional, Dict, Union, List


class ComponentDataModel(BaseModel):
    id: int
    name: str
    EDP: str
    Component: str
    Group: Optional[int] = None
    Quantity: float
    damage_states: int = Field(alias="damage-states")
    median_demand: List[float] = Field(alias="median-demand")
    total_dispesion: List[float] = Field(alias="total-dispersion")
    repair_cost: List[float] = Field(alias="repair-cost")
    cost_dispersion: List[float] = Field(alias="cost-dispersion")
    best_fit: List[Optional[str]] = Field(alias="best-fit", default=None)

    @field_validator('Group', mode="before")
    @classmethod
    def allow_none(cls, v):
        if v is None or v == "":
            return None
        else:
            return v


class CorrelationTreeModel(BaseModel):
    id: int
    dependent_on_item: str = Field(alias="DEPENDANT ON ITEM")
    min_ds: List[str] = Field(alias="MIN DS")


class ItemBase(RootModel):
    root: Dict[str, np.ndarray]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ItemsModel(RootModel):
    root: Dict[int, ItemBase]


class FragilityModel(BaseModel):
    EDP: np.ndarray
    ITEMs: ItemsModel
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DamageStateModel(RootModel):
    root: Dict[int, Dict[int, np.ndarray]]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CostModel(RootModel):
    root: Dict[int, np.ndarray]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimulationModel(RootModel):
    root: Dict[int, CostModel]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FittingModelBase(BaseModel):
    popt: Union[np.ndarray, List]
    pcov: Union[np.ndarray, List]
    multiplier: Optional[float] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FittingParametersModel(RootModel):
    root: Dict[str, FittingModelBase]


class FittedLossModel(RootModel):
    root: Dict[str, np.ndarray]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LossModel(BaseModel):
    loss: Dict[int, Dict[Union[int, str], float]]
    loss_ratio: Dict[int, Dict[Union[int, str], float]]


class SLFModel(BaseModel):
    directionality: Optional[int] = Field(None, alias="Directionality")
    component_type: str = Field(alias="Component-type")
    storey: Optional[Union[int, List[int]]] = Field(None, alias="Storey")
    edp: str
    edp_range: List[float]
    slf: List[float]
    fitting_parameters: Optional[Dict[str, FittingModelBase]] = None


class SLFPGModel(RootModel):
    root: Dict[str, SLFModel]
    model_config = ConfigDict(arbitrary_types_allowed=True)
