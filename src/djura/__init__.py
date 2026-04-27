"""djura — scientific toolkit for earthquake engineering."""
from importlib import import_module

__version__ = "0.1.0"

__citation__ = """\
@software{djura,
  author  = {Shahnazaryan, Davit and Ozsarac, Volkan and O'Reilly, Gerard J.},
  title   = {djura: Scientific Python toolkit for earthquake engineering},
  year    = {2025},
  url     = {https://github.com/davitshahnazaryan3/djura},
  version = {%s}
}
""" % __version__

_SUBMODULES = ("record_selection", "hazard_consistency", "vulnerability_modeller", "slf")


def cite(submodule: str | None = None, style: str = "bibtex", all: bool = False) -> str:
    """Return citation text for djura or a specific submodule.

    Parameters
    ----------
    submodule : str, optional
        One of ``"record_selection"``, ``"hazard_consistency"``,
        ``"vulnerability_modeller"``, ``"slf"``. If ``None`` and ``all`` is
        False, returns the umbrella package citation.
    style : str
        Currently only ``"bibtex"`` is supported.
    all : bool
        If True, returns the umbrella citation plus every submodule citation
        concatenated.
    """
    if style != "bibtex":
        raise ValueError(f"Unsupported style: {style!r}. Only 'bibtex' is supported.")

    if all:
        parts = [__citation__]
        for name in _SUBMODULES:
            mod = import_module(f"djura.{name}")
            parts.append(getattr(mod, "__citation__", ""))
        return "\n".join(p for p in parts if p)

    if submodule is None:
        return __citation__

    if submodule not in _SUBMODULES:
        raise ValueError(
            f"Unknown submodule: {submodule!r}. "
            f"Choose from {_SUBMODULES}."
        )

    mod = import_module(f"djura.{submodule}")
    return getattr(mod, "__citation__", "")
