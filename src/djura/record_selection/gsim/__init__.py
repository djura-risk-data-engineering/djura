# Portions of this subpackage are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ./NOTICE.md for details.
"""
Vendored subset of OpenQuake's `hazardlib.gsim` used by
`djura.record_selection`.

This subpackage is a self-contained slice of the GMPE machinery — the
base class, IMT types, contexts, coefficient tables, constants, and
the model implementations under :mod:`.models`.

Public API is intentionally small; import what you need explicitly,
e.g.::

    from djura.record_selection.gsim.models import BooreEtAl2014
    from djura.record_selection.gsim.imt import SA, PGA
    from djura.record_selection.gsim.contexts import Context
"""
# Submodules are intentionally NOT eagerly imported here: importing
# `.models` pulls in every GMM (some of which require optional deps like
# `shapely`). Import what you need explicitly, e.g.
# `from djura.record_selection.gsim import contexts`.
