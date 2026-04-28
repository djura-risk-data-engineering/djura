from pydantic import BaseModel, field_validator
from typing import List, Dict


class HazardModelSchema(BaseModel):
    s: List[float]
    mafe: List[float]
    s_fit: List[float]
    mafe_fit: List[float]
    method: str
    coef: List[float]
    return_periods: List[float] = None

    class ConfigDict:
        extra = 'forbid'

    @field_validator('coef', mode="before")
    def check_coef_length(cls, coef, values):

        if 'method' in values and values['method'] == "power-law" and \
                len(coef) != 2:

            raise ValueError("For 'power-law' method length of coef must be 2")

        if 'method' in values and values['method'] != 'power-law' and \
                len(coef) != 3:

            raise ValueError("For any method other than 'power-law' "
                             "the length of coef must be 3")

        return coef


class HazardSchema(BaseModel):
    s: List[float]
    poe: List[float]

    class ConfigDict:
        extra = 'forbid'


class HazardBatchSchema(BaseModel):
    root: Dict[str, HazardSchema]

    class ConfigDict:
        arbitrary_types_allowed = True
