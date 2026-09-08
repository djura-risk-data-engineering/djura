# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""
djura.fragility_converter - deprecated alias of :mod:`djura.im_conversion`.

Importing this module emits a :class:`DeprecationWarning` and forwards
everything to :mod:`djura.im_conversion`. The dotted submodule paths
``djura.fragility_converter.ff`` and ``.ff_approximate`` keep resolving to
their new counterparts, so existing imports continue to work unchanged.
Scheduled for removal in djura 3.0.
"""
import sys
import warnings

from ..im_conversion import FF, FFApproximate, __citation__, cite
from ..im_conversion import ff as _ff
from ..im_conversion import ff_approximate as _ff_approximate

warnings.warn(
    "djura.fragility_converter is deprecated and will be removed in djura "
    "3.0; import from djura.im_conversion instead. The install extra is "
    'likewise renamed: use pip install "djura[im_conversion]".',
    DeprecationWarning,
    stacklevel=2,
)

# Make the old dotted paths importable without shipping stub modules.
sys.modules[f"{__name__}.ff"] = _ff
sys.modules[f"{__name__}.ff_approximate"] = _ff_approximate

__all__ = ["FF", "FFApproximate", "cite"]
