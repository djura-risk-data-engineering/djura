# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""djura.hazard_consistency - hazard-consistent IM analysis."""
from .hazard_model import HazardModel
from .hazard_fit import HazardFit
from .hazard_consistency import HazardConsistency

__all__ = ["HazardModel", "HazardFit", "HazardConsistency", "cite"]

__citation__ = (
    "@inproceedings{shahnazaryan2025djuraHC,\n"
    "  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and "
    "O'Reilly, Gerard J.},\n"
    "  title     = {{DJURA Ground Motion Record Selector: "
    "A Software Solution for Earthquake Engineering}},\n"
    "  booktitle = {COMPDYN 2025 Proceedings},\n"
    "  year      = {2025}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the hazard_consistency submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
