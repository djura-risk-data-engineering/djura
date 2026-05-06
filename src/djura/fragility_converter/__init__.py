# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""
djura.fragility_converter - conversion of fragility and vulnerability
models across intensity measures.
"""
from .ff import FF
from .ff_approximate import FFApproximate

__all__ = ["FF", "FFApproximate", "cite"]

__citation__ = (
    "@article{oreilly2025fragility,\n"
    "  author    = {O'Reilly, Gerard J. and Ozsarac, Volkan and "
    "Shahnazaryan, Davit},\n"
    "  title     = {{Conversion of seismic fragility and vulnerability models "
    "to alternative intensity measures for regional risk analysis}},\n"
    "  journal   = {Earthquake Spectra},\n"
    "  year      = {2025},\n"
    "  note      = {Under Review}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the fragility_converter submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
