# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
def make_prediction(
    strength_ratio: float, period: float, case: str, period_c: float
) -> float:
    """This implements the R-mu-T relationship proposed by Guerrini et al 2017

    References
    -------
    Guerrini, G., Graziotti, F., Penna, A.;
    Magenes, G. (2017). Improved evaluation of inelastic displacement
    demands for short-period masonry structures. Earthquake Engineering
    &#38; Structural Dynamics, 46(9), 1411-1430.
    https://doi.org/10.1002/eqe.2862

    Parameters
    -------
    strength_ratio : float
        Strength ratio
    period : float
        Period
    case : str
        See Table II in article, (options: FD, IN, SD)
        FD=flexure-dominated
        IN=intermediate
        SD=shear-dominated
    period_c : float
        Corner period

    Returns
    -------
    float
        Ductility demand

    Raises
    ------
        ValueError if case is not 'fd', 'in', 'sd'
    """

    if strength_ratio <= 1:
        return {"median": strength_ratio}

    if case.lower() == "fd":
        ahyst = 0.7
        period_hyst = 0.055
    elif case.lower() == "in":
        ahyst = 0.2
        period_hyst = 0.030
    elif case.lower() == "sd":
        ahyst = 0.0
        period_hyst = 0.022
    else:
        raise ValueError(
            "Case must be 'fd', 'in', 'sd', for details refer"
            "to https://doi.org/10.1002/eqe.2862")

    b = 2.3
    c = 2.1

    ductility = abs(
        strength_ratio + pow(strength_ratio - 1, c)
        / ((period / period_hyst + ahyst)
           * pow(period / period_c, b)))

    return {"median": ductility}
