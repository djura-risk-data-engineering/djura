"""Tests for the deprecated ``djura.fragility_converter`` alias.

The application was renamed to :mod:`djura.im_conversion` in 2.0.1. The old
import path is kept until 3.0 and must resolve to the very same objects.
"""
import importlib
import sys

import pytest

import djura
from djura import im_conversion


@pytest.fixture
def fresh_alias():
    """Import ``djura.fragility_converter`` with its module cache cleared.

    The deprecation warning fires at module execution, so the alias and its
    forwarded dotted paths have to be evicted for a test to observe it.
    """
    names = [n for n in sys.modules
             if n == "djura.fragility_converter"
             or n.startswith("djura.fragility_converter.")]
    saved = {n: sys.modules.pop(n) for n in names}
    try:
        yield lambda: importlib.import_module("djura.fragility_converter")
    finally:
        sys.modules.update(saved)


def test_import_warns(fresh_alias):
    with pytest.deprecated_call():
        fresh_alias()


def test_alias_exports_are_the_same_objects(fresh_alias):
    with pytest.warns(DeprecationWarning):
        mod = fresh_alias()
    assert mod.FF is im_conversion.FF
    assert mod.FFApproximate is im_conversion.FFApproximate
    assert mod.cite is im_conversion.cite


def test_dotted_paths_still_import(fresh_alias):
    with pytest.warns(DeprecationWarning):
        fresh_alias()
    assert importlib.import_module(
        "djura.fragility_converter.ff") is im_conversion.ff
    assert importlib.import_module(
        "djura.fragility_converter.ff_approximate"
    ) is im_conversion.ff_approximate


def test_cite_accepts_the_old_submodule_name():
    assert djura.cite("fragility_converter") == im_conversion.__citation__
