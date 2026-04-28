def make_prediction(
        dynamic_ductility: float, period: float, period_c: float) -> float:
    """This implements the R-mu-T relationship provided in Annex B of
    Eurocode 8 Part 1

    References
    -------
    CEN. [2004] Eurocode 8: Design of Structures for Earthquake Resistance -
    Part 1: General Rules, Seismic Actions and Rules for Buildings
    (EN 1998-1:2004), Brussels, Belgium.

    Parameters
    -------
    dynamic_ductility : float
        Ductility
    period : float
        Period
    period_c : float
        Corner period

    Returns
    -------
    float
        Strength ratio
    """
    if dynamic_ductility <= 1:
        return {"median": dynamic_ductility}

    if period < period_c:
        strength_ratio = (dynamic_ductility - 1) * (period / period_c) + 1
    else:
        strength_ratio = dynamic_ductility

    return {"median": strength_ratio}
