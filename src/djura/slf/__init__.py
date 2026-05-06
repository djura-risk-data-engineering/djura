# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""djura.slf - storey loss function generation."""

__citation__ = (
    "@inproceedings{shahnazaryan2025slf,\n"
    "  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and "
    "O'Reilly, Gerard J.},\n"
    "  title     = {{The Role of Story Loss Functions in Regional "
    "Seismic Vulnerability Modelling and Risk Assessment}},\n"
    "  year      = {2025}\n"
    "}\n"
)


def cite(style: str = "bibtex") -> str:
    """Return citation text for the slf submodule."""
    if style != "bibtex":
        raise ValueError(
            f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
