# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""djura.record_selection - GCIM-based ground motion record selection."""
from .gcim import GCIM

__all__ = ["GCIM", "cite"]

__citation__ = (
    "@inproceedings{shahnazaryan2025djuraRS,\n"
    "  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and "
    "O'Reilly, Gerard J.},\n"
    "  title     = {{DJURA Ground Motion Record Selector: "
    "A Software Solution for Earthquake Engineering}},\n"
    "  booktitle = {COMPDYN 2025 Proceedings},\n"
    "  year      = {2025}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the record_selection submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
