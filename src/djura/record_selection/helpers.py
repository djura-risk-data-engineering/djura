from numpy import float_, float32, int32, ndarray
from inspect import Parameter, signature


def to_json_serializable(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = to_json_serializable(value)
    elif isinstance(data, list):
        return [to_json_serializable(item) for item in data]
    elif isinstance(data, ndarray):
        return data.tolist()
    elif isinstance(data, float_):
        return float(data)
    elif isinstance(data, float32):
        return float(data)
    elif isinstance(data, int32):
        return float(data)

    return data


def get_func_args(function):
    args = signature(function).parameters

    return args


def filter_args(method, data):

    args = get_func_args(method)

    filtered_data = {}
    for key, param in args.items():
        if key == "self":
            continue

        default_value = param.default

        if key not in data and default_value != Parameter.empty:
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
