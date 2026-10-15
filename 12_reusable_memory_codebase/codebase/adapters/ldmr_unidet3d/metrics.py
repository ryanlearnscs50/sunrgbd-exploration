"""Preserve host metric nesting while making undefined AP explicit in JSON."""
from collections.abc import Mapping
import math
from numbers import Real


def json_metrics(metrics):
    """Return (nested JSON values, undefined metric paths); reject empty results."""
    undefined = []

    def convert(value, path):
        if isinstance(value, Mapping):
            if not value or any(not isinstance(key, str) for key in value):
                raise ValueError('metric groups must be nonempty with string keys')
            return {key: convert(item, path + [key]) for key, item in value.items()}
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'metric must be numeric: {path}')
        if math.isinf(value):
            raise ValueError(f'infinite metric: {path}')
        if math.isnan(value):
            undefined.append(path)
            return None
        return float(value)

    if not isinstance(metrics, Mapping):
        raise ValueError('metrics must be a mapping')
    return convert(metrics, []), undefined
