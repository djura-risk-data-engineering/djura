# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import numpy as np


def make_prediction(dynamic_ductility: float, period: float,
                    period_cc: float, period_c: float) -> float:
    """This implements the R-mu-T relationship proposed by
    Newmark and Hall (1992)

    References
    ------
    Newmark,  N.  M.,  and  Hall,  W.  J.,  1982,
    Earthquake  Spectra  and  Design,
    Earthquake  Engineering  Research Institute, Berkeley, CA

    Parameters
    ------
    dynamic_ductility : float
        Ductility demand
    period : float
        Period
    period_cc : float
        Corner period (called Tc' in article)
    period_c : float
        Corner period (where the A range transitions to the V range)

    Returns
    ------
    float
        Strength ratio
    """

    if dynamic_ductility <= 1:
        return {"median": dynamic_ductility}

    # Set the period values based on Newmark and Hall's spectrum
    period_a = 1. / 33
    period_b = 0.125

    beta = np.log(period / period_a) / np.log(period_b / period_a)

    if period < period_a:
        strength_ratio = 1
    elif period <= period_b:
        strength_ratio = pow(2 * dynamic_ductility - 1, 0.5 * beta)
    elif period <= period_cc:
        strength_ratio = pow(2 * dynamic_ductility - 1, 0.5)
    elif period <= period_c:
        # TODO, unsure here
        # strength_ratio = (period / period_c) * dynamic_ductility
        strength_ratio = (
            period
            * (dynamic_ductility - pow(2 * dynamic_ductility - 1, 0.5)) / (
                period_c - period_cc
            )
            + dynamic_ductility
            - period_c
            * (dynamic_ductility - pow(2 * dynamic_ductility - 1, 0.5)) / (
                period_c - period_cc
            )
        )
    else:
        strength_ratio = dynamic_ductility

    return {"median": strength_ratio}
