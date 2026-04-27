"""djura.slf — storey loss function generation."""

__citation__ = """\
@inproceedings{shahnazaryan2025slf,
  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and O'Reilly, Gerard J.},
  title     = {{The Role of Story Loss Functions in Regional Seismic Vulnerability Modelling and Risk Assessment}},
  year      = {2025}
}
"""


def cite(style: str = "bibtex") -> str:
    """Return citation text for the slf submodule."""
    if style != "bibtex":
        raise ValueError(f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
