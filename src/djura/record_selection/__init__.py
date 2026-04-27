"""djura.record_selection — GCIM-based ground motion record selection."""

__citation__ = """\
@inproceedings{shahnazaryan2025djuraRS,
  author    = {Shahnazaryan, Davit and Ozsarac, Volkan and O'Reilly, Gerard J.},
  title     = {{DJURA Ground Motion Record Selector: A Software Solution for Earthquake Engineering}},
  booktitle = {COMPDYN 2025 Proceedings},
  year      = {2025}
}
"""


def cite(style: str = "bibtex") -> str:
    """Return citation text for the record_selection submodule."""
    if style != "bibtex":
        raise ValueError(f"Unsupported style: {style!r}. Only 'bibtex' is supported.")
    return __citation__
