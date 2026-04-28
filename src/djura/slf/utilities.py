import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import inspect
from scipy.interpolate import interp1d


def get_func_args(function):
    args = inspect.signature(function).parameters

    return args


def to_json_serializable(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = to_json_serializable(value)
    elif isinstance(data, list):
        return [to_json_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float_):
        return float(data)
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.int32):
        return float(data)
    return data


def export_results(filepath: Path, data, filetype: str):
    """Exports results to file

    Parameters
    ----------
    filepath : Path
        Path where to export data to
    data : any
        Data to be stored
    filetype : str
        Filetype, e.g. npy, json, pkl, csv
    """
    if filetype == "json":
        data = to_json_serializable(data)

    if filetype == "npy":
        np.save(f"{filepath}.npy", data)
    elif filetype == "pkl" or filetype == "pickle":
        with open(f"{filepath}.pickle", "wb") as handle:
            pickle.dump(data, handle)
    elif filetype == "json":
        with open(f"{filepath}.json", "w") as json_file:
            json.dump(data, json_file)
    elif filetype == "csv":
        data.to_csv(f"{filepath}.csv", index=False)


def filter_args(method, data):

    args = get_func_args(method)

    filtered_data = {}
    for key, param in args.items():
        if key == "self":
            continue

        default_value = param.default

        if key not in data and default_value != inspect.Parameter.empty:
            filtered_data[key] = default_value
        else:
            if param.annotation is not bool:
                try:
                    _val = float(data[key])
                except (ValueError, TypeError):
                    _val = data[key]

                filtered_data[key] = _val
            else:
                try:
                    filtered_data[key] = data[key].lower() == 'true'
                except AttributeError:
                    filtered_data[key] = data[key]

    return filtered_data


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

    Each argument must have a specific structure
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
    # Initialize list to store the SLFs
    slfs = list()

    # Loop over each provided SLF input
    for arg in args:
        for group in arg.keys():
            edp = arg[group]["edp"]
            loss = arg[group]["slfs"]["mean"]
            loss[loss < 0] = 0.0

            interpolator = interp1d(edp, loss)
            arg[group]["interpolator"] = interpolator

        slfs.append(arg)

    return slfs
