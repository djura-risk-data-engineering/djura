"""Minimal smoke tests for the package skeleton."""
import pytest

import djura
from djura import (
    record_selection,
    hazard_consistency,
    vulnerability_modeller,
    slf,
)


def test_version_present():
    assert isinstance(djura.__version__, str)
    assert djura.__version__.count(".") >= 2


def test_umbrella_citation_returns_bibtex():
    out = djura.cite()
    assert "@software" in out
    assert "djura" in out


@pytest.mark.parametrize("name,mod", [
    ("record_selection", record_selection),
    ("hazard_consistency", hazard_consistency),
    ("vulnerability_modeller", vulnerability_modeller),
    ("slf", slf),
])
def test_submodule_has_citation(name, mod):
    assert hasattr(mod, "__citation__")
    assert mod.__citation__.strip().startswith("@")
    assert djura.cite(name) == mod.__citation__


def test_cite_all_includes_every_submodule():
    out = djura.cite(all=True)
    for key in ("djuraRS", "djuraHC", "shahnazaryan2024nextgen",
                "shahnazaryan2025slf"):
        assert key in out


def test_unknown_submodule_raises():
    with pytest.raises(ValueError):
        djura.cite("nope")


def test_unsupported_style_raises():
    with pytest.raises(ValueError):
        djura.cite(style="apa")
