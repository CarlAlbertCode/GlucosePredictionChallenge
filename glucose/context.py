import numpy as np
import pandas as pd


RAW_FIELDS = ("weight", "height", "age", "age_of_diagnosis")

FEATURE_NAMES = (
    "weight",
    "weight_valid",
    "height",
    "height_valid",
    "age",
    "age_valid",
    "age_of_diagnosis",
    "age_of_diagnosis_valid",
    "diabetes_duration_years",
    "diabetes_duration_valid",
)

def _numeric(values: pd.Series, field: str) -> np.ndarray:
    out = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan, copy=True)
    out[np.isin(out, (-1.0, -2.0))] = np.nan
    if np.isinf(out).any():
        raise ValueError(f"Nonfinite {field}")
    if (np.isfinite(out) & (out < 0)).any():
        raise ValueError(f"Negative {field}")
    return out

def _last_known(times: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(values))
    result = np.full(len(query), np.nan, dtype=np.float64)
    if not len(valid):
        return result
    pos = np.searchsorted(times[valid], query, side="right") - 1
    ok = pos >= 0
    result[ok] = values[valid[pos[ok]]]
    return result

def _encode_values(values: np.ndarray) -> np.ndarray:
    columns = []
    for v in values.T:
        columns.extend((np.nan_to_num(v, nan=0.0), np.isfinite(v).astype(float)))
    age, diagnosis = values[:, 2], values[:, 3]
    valid = np.isfinite(age) & np.isfinite(diagnosis) & (diagnosis <= age)
    columns.extend((np.where(valid, age - diagnosis, 0.0), valid.astype(float)))
    out = np.column_stack(columns).astype(np.float32)
    if out.shape != (len(values), len(FEATURE_NAMES)) or not np.isfinite(out).all():
        raise ValueError("Invalid participant-context block")
    return out

def _encode(frame: pd.DataFrame, origins: pd.Series) -> np.ndarray:
    times = pd.to_datetime(frame.date, errors="raise")
    query = pd.to_datetime(origins, errors="raise")
    if (isinstance(times.dtype, pd.DatetimeTZDtype)
            or isinstance(query.dtype, pd.DatetimeTZDtype)
            or times.isna().any() or query.isna().any()):
        raise ValueError("Recorded timestamps must be nonmissing and timezone-naive")
    time_ns = times.to_numpy("datetime64[ns]").astype(np.int64)
    if len(np.unique(time_ns)) != len(time_ns):
        raise ValueError("Repeated raw timestamps require upstream resolution")
    order = np.argsort(time_ns, kind="stable")
    query_ns = query.to_numpy("datetime64[ns]").astype(np.int64)
    values = np.column_stack([
        _last_known(time_ns[order], _numeric(frame[field], field)[order], query_ns)
        for field in RAW_FIELDS
    ])
    return _encode_values(values)

def weight_coordinates(X: np.ndarray, names: list[str], context: np.ndarray, horizon: int) -> tuple[np.ndarray, tuple[str, ...]]:
    X = np.asarray(X, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    if X.ndim != 2 or context.shape != (len(X), len(FEATURE_NAMES)) or horizon not in (30, 60, 90, 120):
        raise ValueError("Unaligned weight-coordinate inputs")
    lookup = {name: i for i, name in enumerate(names)}
    needed = ["anchor", f"future_insulin_sum_{horizon}", f"future_basal_sum_{horizon}",
              f"future_bolus_sum_{horizon}"]
    for low, high in ((0, 30), (30, 60), (60, 120), (120, 240), (240, 360), (360, 720), (720, 1440)):
        needed.extend((f"insulin_past_{low}_{high}_sum", f"basal_past_{low}_{high}_sum"))
    if not all(name in lookup for name in needed):
        raise ValueError("Frozen base feature schema lacks required therapy coordinates")

    weight = context[:, FEATURE_NAMES.index("weight")]
    valid = context[:, FEATURE_NAMES.index("weight_valid")].astype(bool) & (weight > 0)
    denom = np.where(valid, weight, 1.0)
    past_insulin = np.zeros(len(X), dtype=np.float64)
    past_basal = np.zeros(len(X), dtype=np.float64)
    for low, high in ((0, 30), (30, 60), (60, 120), (120, 240), (240, 360), (360, 720), (720, 1440)):
        past_insulin += X[:, lookup[f"insulin_past_{low}_{high}_sum"]]
        past_basal += X[:, lookup[f"basal_past_{low}_{high}_sum"]]

    raw = np.column_stack((
        X[:, lookup["anchor"]] / denom,
        past_insulin / denom,
        past_basal / denom,
        X[:, lookup[f"future_insulin_sum_{horizon}"]] / denom,
        X[:, lookup[f"future_basal_sum_{horizon}"]] / denom,
        X[:, lookup[f"future_bolus_sum_{horizon}"]] / denom,
        valid.astype(float),
    ))
    raw[~valid, :-1] = 0.0
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite weight-normalized coordinates")
    names_out = (
        "anchor_per_weight",
        "past24h_insulin_per_weight",
        "past24h_basal_per_weight",
        f"future_insulin_{horizon}_per_weight",
        f"future_basal_{horizon}_per_weight",
        f"future_bolus_{horizon}_per_weight",
        "weight_coordinate_valid",
    )
    return raw.astype(np.float32), names_out
