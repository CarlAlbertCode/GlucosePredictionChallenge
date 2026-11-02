import numpy as np
from .data import meal_presence
from .features import EVENTS, GroupArray, _event_bins


FEATURE_NAMES = tuple(
    name
    for minute in range(125, 241, 5)
    for name in (
        *(n for event in EVENTS for n in (f"post120_{event}_{minute}", f"post120_{event}_{minute}_coverage")),
        f"post120_grid_{minute}_observed",
    )
)

def therapy_features(frame, origins_ns):
    """Use only events in (origin+120,origin+240], preserving five-minute bins."""
    required = {"date", "insulin", "basal", "bolus", "carbs", "meal_label"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Nonempty canonical therapy frame required")
    times = frame.date.to_numpy("datetime64[ns]").astype(np.int64)
    if np.any(np.diff(times) <= 0) or frame.date.isna().any():
        raise ValueError("Therapy timestamps must be sorted, unique and nonmissing")
    origins = np.asarray(origins_ns)
    if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
        raise ValueError("Origins must be a one-dimensional nanosecond integer array")
    doses = frame[["insulin", "basal", "bolus", "carbs"]].to_numpy(np.float32)
    if np.isinf(doses).any() or (doses < 0).any():
        raise ValueError("Canonical therapy must be nonnegative or missing")
    # Deliberately synthesize an empty CGM column. Even a caller-supplied CGM
    # column cannot affect event availability, values or timestamps selected.
    values = np.column_stack(
        (np.full(len(frame), np.nan, np.float32), doses, meal_presence(frame.meal_label))
    )
    group = GroupArray(times, values)
    bins, masks, observed = _event_bins(group, origins, count=24, start_offset=120)
    result = np.empty((len(origins), 24, 11), np.float32)
    result[:, :, :10:2], result[:, :, 1:10:2] = bins, masks
    result[:, :, 10] = observed
    return result.reshape(len(origins), len(FEATURE_NAMES))
