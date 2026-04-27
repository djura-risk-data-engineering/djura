# Portions of this file are adapted from the OpenQuake Engine
# (https://github.com/gem/oq-engine), Copyright (C) GEM Foundation,
# licensed under AGPL-3.0-or-later. See ../NOTICE.md (or ../../NOTICE.md
# for files under models/) for full attribution.

import numpy as np
from .const import KNOWN_DISTANCES


def mblg_to_mw_atkinson_boore_87(mag):
    """
    Convert magnitude value from Mblg to Mw using Atkinson and Boore 1987
    conversion equation.

    Implements equation as in line 1656 in hazgridXnga2.f
    """
    return 2.715 - 0.277 * mag + 0.127 * mag * mag


def mblg_to_mw_johnston_96(mag):
    """
    Convert magnitude value from Mblg to Mw using Johnston 1996 conversion
    equation.

    Implements equation as in line 1654 in hazgridXnga2.f
    """
    return 1.14 + 0.24 * mag + 0.0933 * mag * mag


def clip_mean(imt, mean):
    """
    Clip GMPE mean value at 1.5 g for PGA and 3 g for short periods
    (0.02 < T < 0.55)
    """
    if imt.period == 0:
        mean[mean > 0.405] = 0.405

    if 0.02 < imt.period < 0.55:
        mean[mean > 1.099] = 1.099

    return mean


def get_fault_type_dummy_variables(ctx):
    """
    Get fault type dummy variables, see Table 2, pag 107.
    Fault type (Strike-slip, Normal, Thrust/reverse) is
    derived from rake angle.
    Rakes angles within 30 of horizontal are strike-slip,
    angles from 30 to 150 are reverse, and angles from
    -30 to -150 are normal. See paragraph 'Predictor Variables'
    pag 103.
    Note that the 'Unspecified' case is not considered,
    because rake is always given.
    """
    SS = np.zeros_like(ctx.rake)  # strike-slip
    NS = np.zeros_like(ctx.rake)  # normal
    RS = np.zeros_like(ctx.rake)  # reverse
    SS[(np.abs(ctx.rake) <= 30.) | (180. - np.abs(ctx.rake) <= 30.)] = 1.
    RS[(ctx.rake > 30.) & (ctx.rake < 150.)] = 1.
    NS[(ctx.rake > -150.) & (ctx.rake < -30)] = 1.
    return SS, NS, RS


def get_dists(ctx):
    """
    Extract the distance parameters from a context.

    :returns: a dictionary dist_name -> distances
    """
    return {par: dist for par, dist in vars(ctx).items()
            if par in KNOWN_DISTANCES}


class CallableDict(dict):
    def __init__(self, keyfunc=lambda key: key, keymissing=None):
        super().__init__()
        self.keyfunc = keyfunc
        self.keymissing = keymissing

    def add(self, *keys):
        """
        Return a decorator registering a new implementation for the
        CallableDict for the given keys.
        """
        def decorator(func):
            for key in keys:
                self[key] = func
            return func
        return decorator

    def __call__(self, obj, *args, **kw):
        key = self.keyfunc(obj)
        return self[key](obj, *args, **kw)

    def __missing__(self, key):
        if callable(self.keymissing):
            return self.keymissing
        raise KeyError(key)
