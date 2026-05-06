# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import pandas as pd
from scipy.interpolate import interp1d

from ..utilities import (  # noqa: F401 (re-exported)
    to_json_serializable, filter_args)


def aggregate_values(row, columns, key):
    return [row[col] for col in columns if key in col and
            not pd.isnull(row[col])]


def convert_inv(df: pd.DataFrame):
    out = []

    for _, row in df.iterrows():
        median = aggregate_values(row, df.columns, "median-demand")
        dispersion = aggregate_values(row, df.columns, "total-dispersion")
        cost = aggregate_values(row, df.columns, "repair-cost")
        cost_disp = aggregate_values(row, df.columns, "cost-dispersion")
        best_fit = [row[col] for col in df.columns if "best-fit" in col]

        out.append({
            "id": row["id"],
            "EDP": row["EDP"],
            "Component": row["Component"],
            "Group": row["Group"],
            "Quantity": row["Quantity"],
            "damage-states": row["damage-states"],
            "median-demand": median,
            "total-dispersion": dispersion,
            "repair-cost": cost,
            "cost-dispersion": cost_disp,
            "best-fit": best_fit,
        })

    return out


def convert_corr(df: pd.DataFrame):
    out = []

    for _, row in df.iterrows():
        min_ds = aggregate_values(row, df.columns, "MIN DS")

        out.append({
            "ITEM": row["ITEM"],
            "DEPENDANT ON ITEM": row["DEPENDANT ON ITEM"],
            "MIN DS": min_ds,
        })

    return out


def slf_aggregator(*args) -> list:
    """Aggregates input SLFs into a list supported by the loss assessment
    module and creates interpolation functions for the EDP vs Loss
    relationships

    Each argument must have a specific structure::

        {
            'group name': {
                'Directionality': null,
                'Storey': null,
                'edp': 'edp_name',
                'edp_range': list(),
                'slf': list()
            }
        }
    """
    slfs = list()

    for arg in args:
        for group in arg.keys():
            edp = arg[group]["edp"]
            loss = arg[group]["slfs"]["mean"]
            loss[loss < 0] = 0.0

            interpolator = interp1d(edp, loss)
            arg[group]["interpolator"] = interpolator

        slfs.append(arg)

    return slfs
