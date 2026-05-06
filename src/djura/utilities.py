# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
import pickle
import json
import inspect
from pathlib import Path
from typing import Union

import numpy as np


def to_json_serializable(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = to_json_serializable(value)
    elif isinstance(data, list):
        return [to_json_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.int32):
        return float(data)
    return data


def get_func_args(function):
    return inspect.signature(function).parameters


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
                    if param.annotation is int:
                        _val = int(data[key])
                    else:
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


def read_json(filename: Union[Path, dict]):
    if isinstance(filename, (Path, str)):
        filename = Path(filename)

        with open(filename) as f:
            filename = json.load(f)

    return filename


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
