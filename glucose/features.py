"""Bounded causal features and lazy neural windows; exact-time endpoint labels.

Only recorded past CGM is used. Future features contain supplied insulin/meals,
and the per-horizon mask prevents a tree from seeing events beyond its endpoint.
No identity/source key is exposed as a predictive feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEYS, canonicalize, meal_presence

HORIZONS = (30, 60, 90, 120)
MINUTE_NS = 60_000_000_000
EVENTS = ("insulin", "basal", "bolus", "carbs", "meal")
HISTORY_CHANNELS = (
    "cgm_scaled",
    "cgm_delta",
    "cgm_diff",
    "cgm_valid",
    "cgm_age",
    "insulin",
    "insulin_valid",
    "basal",
    "basal_valid",
    "bolus",
    "bolus_valid",
    "carbs_scaled",
    "carbs_valid",
    "meal",
    "meal_valid",
    "clock_sin",
    "clock_cos",
    "offset",
    "grid_observed",
    "insulin_memory_240",
    "insulin_memory_360",
    "carb_memory_60",
    "carb_memory_180",
)
FUTURE_CHANNELS = (
    "insulin",
    "insulin_valid",
    "basal",
    "basal_valid",
    "bolus",
    "bolus_valid",
    "carbs_scaled",
    "carbs_valid",
    "meal",
    "meal_valid",
    "clock_sin",
    "clock_cos",
    "offset",
    "grid_observed",
)


@dataclass
class GroupArray:
    times: np.ndarray
    values: np.ndarray  # CGM, insulin, basal, bolus, carbs, meal; raw-unit NaNs retained
    _cache: dict = field(default_factory=dict, repr=False)

    def valid_cgm_indexes(self) -> np.ndarray:
        if "cgm_valid" not in self._cache:
            self._cache["cgm_valid"] = np.flatnonzero(np.isfinite(self.values[:, 0]))
        return self._cache["cgm_valid"]

    def valid_cgm_times(self) -> np.ndarray:
        if "cgm_valid_times" not in self._cache:
            self._cache["cgm_valid_times"] = self.times[self.valid_cgm_indexes()]
        return self._cache["cgm_valid_times"]

    def event_prefixes(self, channel: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("event_prefix", channel)
        if key not in self._cache:
            raw = self.values[:, channel]
            self._cache[key] = (_prefix(np.nan_to_num(raw)), _prefix(np.isfinite(raw)))
        return self._cache[key]

    def event_rows(self) -> np.ndarray:
        if "event_rows" not in self._cache:
            self._cache["event_rows"] = _prefix(np.isfinite(self.values[:, 1:]).any(axis=1))
        return self._cache["event_rows"]

    def basis_prefixes(self, channel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("basis_prefix", channel)
        if key not in self._cache:
            minutes = (self.times - self.times[0]) / MINUTE_NS
            dose = np.nan_to_num(self.values[:, channel].astype(np.float64))
            self._cache[key] = (
                self.event_prefixes(channel)[0],
                _prefix(dose * minutes),
                _prefix(dose * minutes**2),
            )
        return self._cache[key]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "GroupArray":
        values = frame[["CGM", "insulin", "basal", "bolus", "carbs"]].to_numpy(np.float32)
        values = np.column_stack((values, meal_presence(frame.meal_label)))
        return cls(frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64), values)

    def save(self, directory: Path, precompute_prefixes: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "times.npy", self.times)
        np.save(directory / "values.npy", self.values)
        if precompute_prefixes:
            np.save(directory / "cgm_valid.npy", self.valid_cgm_indexes())
            np.save(directory / "cgm_valid_times.npy", self.valid_cgm_times())
            np.save(directory / "event_rows.npy", self.event_rows())
            for channel in range(1, 6):
                sums, counts = self.event_prefixes(channel)
                np.save(directory / f"event_sum_{channel}.npy", sums)
                np.save(directory / f"event_count_{channel}.npy", counts)
            for channel in (1, 4):
                _, first, second = self.basis_prefixes(channel)
                np.save(directory / f"basis_first_{channel}.npy", first)
                np.save(directory / f"basis_second_{channel}.npy", second)

    @classmethod
    def mmap(cls, directory: Path) -> "GroupArray":
        group = cls(
            np.load(directory / "times.npy", mmap_mode="r"), np.load(directory / "values.npy", mmap_mode="r")
        )
        if (directory / "event_rows.npy").exists():
            for name in ("cgm_valid", "cgm_valid_times", "event_rows"):
                group._cache[name] = np.load(directory / (name + ".npy"), mmap_mode="r")
            for channel in range(1, 6):
                group._cache[("event_prefix", channel)] = (
                    np.load(directory / f"event_sum_{channel}.npy", mmap_mode="r"),
                    np.load(directory / f"event_count_{channel}.npy", mmap_mode="r"),
                )
            for channel in (1, 4):
                group._cache[("basis_prefix", channel)] = (
                    group._cache[("event_prefix", channel)][0],
                    np.load(directory / f"basis_first_{channel}.npy", mmap_mode="r"),
                    np.load(directory / f"basis_second_{channel}.npy", mmap_mode="r"),
                )
        return group


@dataclass
class Examples:
    X: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame
    anchor: np.ndarray
    feature_names: list[str]
    horizon_feature_mask: np.ndarray
    window_store: "WindowStore"

    def for_horizon(self, horizon: int) -> np.ndarray:
        return self.X[:, self.horizon_feature_mask[HORIZONS.index(horizon)]]


def _exact_values(group: GroupArray, query: np.ndarray, channel: int = 0) -> np.ndarray:
    indexes = np.searchsorted(group.times, query)
    safe = np.minimum(indexes, len(group.times) - 1)
    found = (indexes < len(group.times)) & (group.times[safe] == query)
    return np.where(found, group.values[safe, channel], np.nan)


def _anchor(group: GroupArray, origins: np.ndarray, fallback: float) -> tuple[np.ndarray, np.ndarray]:
    valid = group.valid_cgm_indexes()
    if not len(valid):
        return np.full(len(origins), fallback), np.full(len(origins), 1440.0)
    positions = np.searchsorted(group.valid_cgm_times(), origins, side="right") - 1
    exists = positions >= 0
    row = valid[np.maximum(positions, 0)]
    anchor = np.where(exists, group.values[row, 0], fallback)
    age = np.where(exists, (origins - group.times[row]) / MINUTE_NS, 1440.0)
    return anchor, np.minimum(age, 1440)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(values, dtype=np.float64)]


def _event_bins(
    group: GroupArray, origins: np.ndarray, count: int = 24, start_offset: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints = origins[:, None] + (start_offset + np.arange(count + 1)[None, :] * 5) * MINUTE_NS
    indexes = np.searchsorted(group.times, endpoints, side="right")
    bins = np.zeros((len(origins), count, len(EVENTS)), dtype=np.float32)
    masks = np.zeros_like(bins)
    # Availability uses event fields alone. CGM-only future rows cannot influence
    # features, even by revealing their presence/cadence through event masks.
    event_rows = group.event_rows()
    rows = np.diff(event_rows[indexes], axis=1)
    for channel in range(5):
        sums, counts = group.event_prefixes(channel + 1)
        bins[:, :, channel] = np.diff(sums[indexes], axis=1)
        masks[:, :, channel] = np.divide(
            np.diff(counts[indexes], axis=1), rows, out=np.zeros_like(rows, dtype=float), where=rows > 0
        )
    return bins, masks, rows > 0


def _basis(group: GroupArray, origins: np.ndarray, channel: int, support: int) -> np.ndarray:
    """Vectorized polynomial response basis, using ONLY dose timestamps <= query."""
    query = (origins - group.times[0]) / MINUTE_NS
    left = np.searchsorted(group.times, origins - support * MINUTE_NS, side="right")
    right = np.searchsorted(group.times, origins, side="right")
    p0, p1, p2 = group.basis_prefixes(channel)
    a = 1 - query / support
    result = (
        a * a * (p0[right] - p0[left])
        + 2 * a / support * (p1[right] - p1[left])
        + (p2[right] - p2[left]) / support**2
    )
    return np.maximum(result, 0).astype(np.float32)


def _group_features(
    group: GroupArray, origins: np.ndarray, fallback: float, include_future: bool
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    columns, names, min_horizons = [], [], []

    def add(name: str, value: np.ndarray, min_horizon: int = 0) -> None:
        columns.append(np.nan_to_num(value, nan=0, posinf=0, neginf=0).astype(np.float32))
        names.append(name)
        min_horizons.append(min_horizon)

    anchors, ages = _anchor(group, origins, fallback)
    add("anchor", anchors)
    add("cgm_observation_age_minutes", ages)
    valid_cgm_times = group.times[np.isfinite(group.values[:, 0])]
    add("anchor_available", np.searchsorted(valid_cgm_times, origins, side="right") > 0)
    for lag in range(0, 121, 5):
        glucose = _exact_values(group, origins - lag * MINUTE_NS)
        add(f"cgm_lag_{lag}", glucose)
        add(f"cgm_lag_{lag}_valid", np.isfinite(glucose))
    raw = group.values[:, 0].astype(float)
    valid = np.isfinite(raw)
    y0 = np.nan_to_num(raw)
    time = (group.times - group.times[0]) / MINUTE_NS
    sums = [_prefix(v) for v in (valid, y0, y0 * y0, valid * time, valid * time * time, y0 * time)]
    right = np.searchsorted(group.times, origins, side="right")
    slopes = {}
    for width in (15, 30, 60, 120, 360, 1440):
        left = np.searchsorted(group.times, origins - width * MINUTE_NS, side="right")
        count, sy, syy, st, stt, sty = (p[right] - p[left] for p in sums)
        mean = np.divide(sy, count, out=np.zeros(len(origins)), where=count > 0)
        if width != 15:
            variance = np.divide(syy, count, out=np.zeros(len(origins)), where=count > 0) - mean**2
            ranges = np.zeros(len(origins))
            for i, (start, stop) in enumerate(zip(left, right)):
                observed = raw[start:stop][valid[start:stop]]
                if len(observed):
                    ranges[i] = observed.max() - observed.min()
            add(f"cgm_mean_{width}", mean)
            add(f"cgm_sd_{width}", np.sqrt(np.maximum(variance, 0)))
            add(f"cgm_range_{width}", ranges)
            add(f"cgm_coverage_{width}", np.minimum(count / (width / 5), 1))
        if width in (15, 30, 60):
            denominator = count * stt - st * st
            slopes[width] = np.divide(
                count * sty - st * sy, denominator, out=np.zeros(len(origins)), where=denominator > 1e-9
            )
            add(f"cgm_slope_{width}", slopes[width])
            add(f"cgm_slope_{width}_valid", count >= 2)
    add("cgm_slope_change_15_60", slopes[15] - slopes[60])
    for event, channel in zip(EVENTS, range(1, 6)):
        values = group.values[:, channel]
        sums_event, count_event = group.event_prefixes(channel)
        for low, high in zip((0, 30, 60, 120, 240, 360, 720), (30, 60, 120, 240, 360, 720, 1440)):
            left = np.searchsorted(group.times, origins - high * MINUTE_NS, side="right")
            stop = np.searchsorted(group.times, origins - low * MINUTE_NS, side="right")
            add(f"{event}_past_{low}_{high}_sum", sums_event[stop] - sums_event[left])
            add(
                f"{event}_past_{low}_{high}_coverage",
                np.minimum((count_event[stop] - count_event[left]) / ((high - low) / 5), 1),
            )
        positive = np.flatnonzero(np.isfinite(values) & (values > 0))
        if len(positive):
            position = np.searchsorted(group.times[positive], origins, side="right") - 1
            elapsed = np.where(
                position >= 0, (origins - group.times[positive[np.maximum(position, 0)]]) / MINUTE_NS, 1440
            )
        else:
            elapsed = np.full(len(origins), 1440)
        add(f"{event}_recency_minutes", np.minimum(elapsed, 1440))
        for run_type, run_condition in (("missing", np.isfinite(values)), ("zero", values != 0)):
            stop_run = np.flatnonzero(run_condition)
            position = (
                np.searchsorted(group.times[stop_run], origins, side="right") - 1 if len(stop_run) else None
            )
            elapsed = np.full(len(origins), 1440.0)
            if position is not None:
                elapsed = np.where(
                    position >= 0,
                    (origins - group.times[stop_run[np.maximum(position, 0)]]) / MINUTE_NS,
                    1440,
                )
            add(f"{event}_{run_type}_run_minutes", np.minimum(elapsed, 1440))
    for channel, event, supports in ((1, "insulin", (240, 360)), (4, "carb", (60, 180))):
        for support in supports:
            add(f"{event}_memory_{support}", _basis(group, origins, channel, support))
            left = np.searchsorted(group.times, origins - support * MINUTE_NS, side="right")
            counts = group.event_prefixes(channel)[1]
            add(
                f"{event}_memory_{support}_coverage",
                np.minimum((counts[right] - counts[left]) / (support / 5), 1),
            )
    clock = (origins % (1440 * MINUTE_NS)) / (1440 * MINUTE_NS) * 2 * np.pi
    add("clock_sin", np.sin(clock))
    add("clock_cos", np.cos(clock))
    if include_future:
        bins, masks, observed = _event_bins(group, origins)
        for j in range(24):
            end = (j + 1) * 5
            for k, event in enumerate(EVENTS):
                add(f"future_{event}_{end}", bins[:, j, k], end)
                add(f"future_{event}_{end}_coverage", masks[:, j, k], end)
            add(f"future_grid_{end}_observed", observed[:, j], end)
        for horizon in HORIZONS:
            n = horizon // 5
            for k, event in enumerate(EVENTS):
                add(f"future_{event}_sum_{horizon}", bins[:, :n, k].sum(axis=1), horizon)
                add(f"future_{event}_coverage_{horizon}", masks[:, :n, k].mean(axis=1), horizon)
            # Project ONLY historical events plus the supplied event prefix.
            for channel, event, supports in ((1, "insulin", (240, 360)), (4, "carb", (60, 180))):
                for support in supports:
                    historical = _historical_projection(group, origins, channel, support, horizon)
                    k = 0 if channel == 1 else 3
                    future_age = horizon - np.arange(1, n + 1) * 5
                    remaining = np.maximum(0, 1 - future_age / support) ** 2
                    add(
                        f"{event}_projected_memory_{support}_h{horizon}",
                        historical + (bins[:, :n, k] * remaining).sum(axis=1),
                        horizon,
                    )
    mask = np.asarray(min_horizons)[None, :] <= np.asarray(HORIZONS)[:, None]
    return np.column_stack(columns), anchors[:, None].astype(np.float32), names, mask


def _historical_projection(group, origins, channel, support, horizon):
    """Evaluate an origin-truncated historical dose stream at a later endpoint."""
    query = (origins - group.times[0]) / MINUTE_NS + horizon
    left = np.searchsorted(group.times, origins + (horizon - support) * MINUTE_NS, side="right")
    right = np.searchsorted(group.times, origins, side="right")
    left = np.minimum(left, right)
    p0, p1, p2 = group.basis_prefixes(channel)
    a = 1 - query / support
    return np.maximum(
        0,
        a * a * (p0[right] - p0[left])
        + 2 * a / support * (p1[right] - p1[left])
        + (p2[right] - p2[left]) / support**2,
    )


def _balanced_positions(keys: pd.DataFrame, maximum: int) -> np.ndarray:
    groups = list(keys.groupby(["source_file", "id"], sort=False).indices.values())
    if maximum >= len(keys):
        return np.arange(len(keys))
    if maximum < 1:
        raise ValueError("max_examples must be positive")
    if maximum < len(groups):
        selected = np.linspace(0, len(groups) - 1, maximum, dtype=int)
        return np.sort(np.asarray([groups[g][len(groups[g]) // 2] for g in selected]))
    quotas = np.zeros(len(groups), dtype=int)
    remaining = maximum
    while remaining:
        available = np.asarray([len(g) for g in groups]) - quotas
        active = np.flatnonzero(available > 0)
        allocations = np.minimum(available[active], max(1, remaining // len(active)))
        for g, allocation in zip(active, allocations):
            allocation = min(int(allocation), remaining)
            quotas[g] += allocation
            remaining -= allocation
            if not remaining:
                break
    return np.sort(
        np.concatenate(
            [
                indexes[np.linspace(0, len(indexes) - 1, quota, dtype=int)]
                for indexes, quota in zip(groups, quotas)
                if quota
            ]
        )
    )


def build_examples(
    frame: pd.DataFrame,
    origins: pd.DataFrame | None = None,
    stride_minutes: int = 15,
    max_examples: int | None = 200_000,
    fallback_anchor: float = 120.0,
    include_future: bool = True,
    shard_dir: str | Path | None = None,
) -> Examples:
    """Build bounded tabular anchors and share raw arrays with a lazy window store.

    Explicit origins preserve their supplied order and are never sampled. Default
    generated origins are subject-balanced and temporally spread if capped.
    ``fallback_anchor`` is a frozen training-fold value, not fitted on this input.
    ``include_future`` assumes these event streams are actually supplied at inference.
    """
    if not np.isfinite(fallback_anchor) or fallback_anchor < 0:
        raise ValueError("fallback_anchor must be finite and nonnegative")
    if stride_minutes <= 0:
        raise ValueError("stride_minutes must be positive")
    clean = canonicalize(frame)
    groups = {key: GroupArray.from_frame(g) for key, g in clean.groupby(["source_file", "id"], sort=False)}
    if origins is None:
        parts = []
        for _, group in clean.groupby(["source_file", "id"], sort=False):
            candidate = group.loc[group.CGM.notna(), KEYS]
            candidate_times = candidate.date.astype("int64").to_numpy()
            # Endpoint availability only chooses training examples; it never
            # supplies any target value to predictive features.
            group_array = groups[(str(group.source_file.iloc[0]), str(group.id.iloc[0]))]
            label_available = np.column_stack(
                [np.isfinite(_exact_values(group_array, candidate_times + h * MINUTE_NS)) for h in HORIZONS]
            ).any(axis=1)
            candidate = candidate.loc[label_available]
            ns = candidate.date.astype("int64").to_numpy()
            if len(ns):
                slot = (ns - ns[0]) // (stride_minutes * MINUTE_NS)
                parts.append(candidate.loc[np.r_[True, np.diff(slot) > 0], KEYS])
        keys = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEYS)
        if max_examples is not None:
            keys = keys.iloc[_balanced_positions(keys, max_examples)].reset_index(drop=True)
    else:
        keys = origins[KEYS].copy().reset_index(drop=True)
        keys["date"] = pd.to_datetime(keys["date"])
        if keys[KEYS].isna().any().any():
            raise ValueError("Origins require unique nonmissing source/id/date keys")
        if isinstance(keys.date.dtype, pd.DatetimeTZDtype):
            raise ValueError("Origins must use recorded naive timestamps")
        keys["date"] = keys.date.astype("datetime64[ns]")
        for key in ("source_file", "id"):
            keys[key] = keys[key].astype(str)
        if keys.duplicated(KEYS).any():
            raise ValueError("Origins require unique nonmissing source/id/date keys")
    if not len(keys):
        raise ValueError("No forecast origins available")
    X, anchors, names, mask = None, np.zeros((len(keys), 1), np.float32), None, None
    y = np.full((len(keys), 4), np.nan, np.float32)
    assignments = np.empty(len(keys), dtype=np.int32)
    group_order = []
    origin_groups = keys.groupby(["source_file", "id"], sort=False).indices.items()
    for group_index, (key, positions) in enumerate(origin_groups):
        if key not in groups:
            raise ValueError(f"Origins reference unknown source/participant: {key}")
        group = groups[key]
        times = keys.iloc[positions].date.astype("int64").to_numpy()
        current_X, anchors[positions], current_names, current_mask = _group_features(
            group, times, fallback_anchor, include_future
        )
        if X is None:
            X = np.empty((len(keys), current_X.shape[1]), np.float32)
            names, mask = current_names, current_mask
        X[positions] = current_X
        for h, horizon in enumerate(HORIZONS):
            y[positions, h] = _exact_values(group, times + horizon * MINUTE_NS)
        assignments[positions] = group_index
        if shard_dir is not None:
            directory = Path(shard_dir) / f"group_{group_index:05d}"
            group.save(directory)
            group = GroupArray.mmap(directory)
        group_order.append(group)
    store = WindowStore(
        group_order,
        assignments,
        keys.date.astype("int64").to_numpy(),
        anchors,
        y,
        include_future,
        fallback_anchor,
    )
    return Examples(X, y, keys, anchors, names, mask, store)


class WindowStore:
    """Share compact group arrays; generate one 180+24-token window on access."""

    history_channels = HISTORY_CHANNELS
    future_channels = FUTURE_CHANNELS

    def __init__(self, groups, assignments, origins, anchors, y, include_future, fallback):
        self.groups, self.assignments, self.origins = groups, assignments, origins
        self.anchors, self.y = anchors, y
        self.include_future, self.fallback = include_future, fallback

    def neural_dataset(self, indices=None):
        return NeuralDataset(self, indices)

    def item(self, index: int) -> dict[str, np.ndarray]:
        group = self.groups[self.assignments[index]]
        origin, anchor = self.origins[index], self.anchors[index, 0]
        history_times = origin + np.arange(-179, 1, dtype=np.int64) * 5 * MINUTE_NS
        raw = np.column_stack([_exact_values(group, history_times, c) for c in range(6)])
        grid_indexes = np.searchsorted(group.times, history_times)
        grid_safe = np.minimum(grid_indexes, len(group.times) - 1)
        observed = (grid_indexes < len(group.times)) & (group.times[grid_safe] == history_times)
        _, age = _anchor(group, history_times, self.fallback)
        glucose_valid = np.isfinite(raw[:, 0])
        glucose = np.nan_to_num(raw[:, 0])
        difference = np.r_[0, np.diff(glucose)] / 100
        difference[~(glucose_valid & np.r_[False, glucose_valid[:-1]])] = 0
        history = [
            glucose / 100,
            np.where(glucose_valid, (glucose - anchor) / 100, 0),
            difference,
            glucose_valid,
            age / 1440,
        ]
        event_bins, event_masks, _ = _event_bins(group, np.asarray([origin]), count=180, start_offset=-900)
        for channel in range(5):
            scale = 50 if channel == 3 else 1
            history.extend([event_bins[0, :, channel] / scale, event_masks[0, :, channel]])
        clock = (history_times % (1440 * MINUTE_NS)) / (1440 * MINUTE_NS) * 2 * np.pi
        history.extend([np.sin(clock), np.cos(clock), np.arange(-179, 1) / 180, observed])
        for channel, supports in ((1, (240, 360)), (4, (60, 180))):
            for support in supports:
                scale = 50 if channel == 4 else 1
                history.append(_basis(group, history_times, channel, support) / scale)
        future_times = origin + np.arange(1, 25) * 5 * MINUTE_NS
        if self.include_future:
            bins, masks, future_observed = _event_bins(group, np.asarray([origin]))
            bins, masks, future_observed = bins[0], masks[0], future_observed[0]
        else:
            bins, masks, future_observed = np.zeros((24, 5)), np.zeros((24, 5)), np.zeros(24)
        future = []
        for channel in range(5):
            scale = 50 if channel == 3 else 1
            future.extend([bins[:, channel] / scale, masks[:, channel]])
        clock = (future_times % (1440 * MINUTE_NS)) / (1440 * MINUTE_NS) * 2 * np.pi
        future.extend([np.sin(clock), np.cos(clock), np.arange(1, 25) / 24, future_observed])
        targets = self.y[index]
        return {
            "history": np.column_stack(history).astype(np.float32),
            "future": np.column_stack(future).astype(np.float32),
            "history_padding_mask": history_times < group.times[0],
            "anchor": self.anchors[index].copy(),
            "target": np.nan_to_num(targets - anchor).astype(np.float32),
            "target_mask": np.isfinite(targets),
            "y": targets.copy(),
        }


class NeuralDataset:
    """PyTorch-compatible map-style dataset without eager window copies."""

    history_channels = HISTORY_CHANNELS
    future_channels = FUTURE_CHANNELS

    def __init__(self, store: WindowStore, indices=None):
        self.store = store
        self.indices = np.arange(len(store.origins)) if indices is None else np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.store.item(int(self.indices[index]))
