import numpy as np
from .features import HORIZONS


COVERAGE_MINIMUM = 0.9

PAST_INTERVALS = ((0, 30), (30, 60), (60, 120), (120, 240), (240, 360), (360, 720), (720, 1440))

COORDINATES = (
    "recent_basal_relative",
    "future_basal_prefix_relative",
    "future_basal_half_change_relative",
    "future_basal_end_minus_recent_relative",
    "future_bolus_fraction_daily",
    "future_insulin_fraction_daily",
)

EXTRA_NAMES = (
    "context_basal24h_coverage",
    "context_insulin24h_coverage",
    *(name for coordinate in COORDINATES for name in (coordinate, coordinate + "_valid")),
)

def relative_therapy_features(X, names, masks, *, horizon_minutes=120):
    """Return14 fixed coordinates without altering X or consuming labels/identities.

    No coverage correction estimates missing doses. Known trailing24h basal sum
    divided by288 nominal five-minute bins defines the basal scale. Known total
    insulin over the same period defines a daily-dose scale. A missing/zero scale
    or insufficient coverage gives value0 AND invalid0, with no epsilon/clipping.
    """
    X, masks = np.asarray(X), np.asarray(masks, dtype=bool)
    names = list(names)
    if (
        horizon_minutes not in HORIZONS
        or X.ndim != 2
        or X.shape[1] != len(names)
        or len(names) != len(set(names))
        or masks.shape != (4, len(names))
    ):
        raise ValueError("Causal features require aligned unique names and four horizon masks")
    lookup, head = {name: i for i, name in enumerate(names)}, HORIZONS.index(horizon_minutes)

    def column(name, coverage=False):
        index = lookup[name]
        if not masks[head, index]:
            raise ValueError(f"Feature {name} is forbidden at the current endpoint")
        value = np.asarray(X[:, index], dtype=np.float64)
        if not np.isfinite(value).all() or (value < 0).any() or (coverage and (value > 1).any()):
            raise ValueError("Dose amounts and coverage must be finite nonnegative causal inputs")
        return value

    past_total, past_coverage = {}, {}
    for event in ("basal", "insulin"):
        past_total[event], past_coverage[event] = np.zeros(len(X)), np.zeros(len(X))
        for low, high in PAST_INTERVALS:
            past_total[event] += column(f"{event}_past_{low}_{high}_sum")
            past_coverage[event] += column(f"{event}_past_{low}_{high}_coverage", True) * ((high - low) / 5)
        past_coverage[event] /= 288
    basal_scale = past_total["basal"] / 288
    insulin_scale = past_total["insulin"]
    basal_valid = (past_coverage["basal"] >= COVERAGE_MINIMUM) & np.isfinite(basal_scale) & (basal_scale > 0)
    insulin_valid = (
        (past_coverage["insulin"] >= COVERAGE_MINIMUM) & np.isfinite(insulin_scale) & (insulin_scale > 0)
    )
    recent = column("basal_past_0_30_sum") / 6
    recent_valid = column("basal_past_0_30_coverage", True) >= COVERAGE_MINIMUM
    n = horizon_minutes // 5
    future, coverage = {}, {}
    for event in ("basal", "bolus", "insulin"):
        future[event] = np.column_stack(
            [column(f"future_{event}_{end}") for end in range(5, horizon_minutes + 1, 5)]
        )
        coverage[event] = np.column_stack(
            [column(f"future_{event}_{end}_coverage", True) for end in range(5, horizon_minutes + 1, 5)]
        )
    future_basal_valid = coverage["basal"].mean(axis=1) >= COVERAGE_MINIMUM
    half_valid = (coverage["basal"][:, : n // 2].mean(axis=1) >= COVERAGE_MINIMUM) & (
        coverage["basal"][:, n // 2 :].mean(axis=1) >= COVERAGE_MINIMUM
    )
    end_valid = coverage["basal"][:, -6:].mean(axis=1) >= COVERAGE_MINIMUM
    values = (
        recent,
        future["basal"].mean(axis=1),
        future["basal"][:, n // 2 :].mean(axis=1) - future["basal"][:, : n // 2].mean(axis=1),
        future["basal"][:, -6:].mean(axis=1) - recent,
        future["bolus"].sum(axis=1),
        future["insulin"].sum(axis=1),
    )
    valid = (
        basal_valid & recent_valid,
        basal_valid & future_basal_valid,
        basal_valid & future_basal_valid & half_valid,
        basal_valid & future_basal_valid & recent_valid & end_valid,
        insulin_valid & (coverage["bolus"].mean(axis=1) >= COVERAGE_MINIMUM),
        insulin_valid & (coverage["insulin"].mean(axis=1) >= COVERAGE_MINIMUM),
    )
    columns = [past_coverage["basal"], past_coverage["insulin"]]
    for index, (value, available) in enumerate(zip(values, valid, strict=True)):
        denominator = basal_scale if index < 4 else insulin_scale
        coordinate = np.divide(value, denominator, out=np.zeros(len(X)), where=available)
        columns.extend((coordinate, available.astype(float)))
    extra = np.column_stack(columns).astype(np.float32)
    if extra.shape != (len(X), 14) or not np.isfinite(extra).all():
        raise ValueError("Fixed ratios must fit finite float32 without clipping or denominator flooring")
    return extra
