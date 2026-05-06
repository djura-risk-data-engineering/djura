# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""
djura.edp_im - EDP-IM relationship prediction using machine learning models.
"""
from .predict import EDPIMModel, EDPIMInfillModel, EDPIMIsolModel
from .predict import edp_im, edp_im_infill, edp_im_isol
from .predict import BackboneModel

__all__ = [
    "EDPIMModel", "EDPIMInfillModel", "EDPIMIsolModel",
    "BackboneModel",
    "edp_im", "edp_im_infill", "edp_im_isol",
    "cite",
]

__citation__ = (
    "@article{shahnazaryan2024nextgen,\n"
    "  author  = {Shahnazaryan, Davit and O'Reilly, Gerard J.},\n"
    "  title   = {{Next-generation non-linear and collapse prediction models "
    "for short- to long-period systems via machine learning methods}},\n"
    "  journal = {Engineering Structures},\n"
    "  volume  = {306},\n"
    "  pages   = {117801},\n"
    "  year    = {2024},\n"
    "  doi     = {10.1016/j.engstruct.2024.117801}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the edp_im submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
