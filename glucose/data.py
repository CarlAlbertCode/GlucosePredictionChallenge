"""Explicit, audited input contract for harmonized five-minute challenge inputs.

Values -1/-2 denote missing data. Zero events are observed zeroes; missing events
stay missing. Insulin totals are never reconstructed from basal/bolus columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEYS = ["source_file", "id", "date"]
NUMERIC = ["CGM", "insulin", "basal", "bolus", "carbs"]
CANONICAL = KEYS + NUMERIC + ["meal_label"]
SENTINELS = (-1.0, -2.0)


@dataclass(frozen=True)
class InputContract:
    """Column names and units supplied by a source dictionary, never guessed.

    ``columns`` maps canonical names to source names. Basal U/hour conversion is
    specifically pump-rate conversion into U per five-minute bin. It must never
    be used for long-acting injection doses. Date values are recorded naive time.
    """

    columns: Mapping[str, str] = field(default_factory=lambda: {c: c for c in CANONICAL})
    glucose_units: str = "mg/dL"
    insulin_units: str = "U/bin"
    basal_units: str = "U/bin"
    bolus_units: str = "U/injection"
    carb_units: str = "grams"
    invalid_negative_events: str = "reject"

    def __post_init__(self) -> None:
        if self.glucose_units not in {"mg/dL", "mmol/L"}:
            raise ValueError("Explicit glucose_units must be mg/dL or mmol/L")
        if self.insulin_units != "U/bin" or self.basal_units not in {"U/bin", "U/hour"}:
            raise ValueError("Delivered insulin must be U/bin; pump basal may explicitly be U/hour")
        if self.bolus_units not in {"U/injection", "U/bin"} or self.carb_units != "grams":
            raise ValueError("Bolus must be delivered units and carbohydrate must be grams")
        if self.invalid_negative_events not in {"reject", "mask"}:
            raise ValueError("invalid_negative_events must be reject or mask")
        unknown = set(self.columns) - set(CANONICAL)
        if unknown:
            raise ValueError(f"Unknown canonical fields: {sorted(unknown)}")
        if len(set(self.columns.values())) != len(self.columns):
            raise ValueError("A source column cannot map to multiple canonical fields")


def canonicalize(frame: pd.DataFrame, contract: InputContract | None = None) -> pd.DataFrame:
    """Project the allowlist, mask sentinels, validate keys/units, and sort groups."""
    contract = contract or InputContract()
    missing = [c for c in KEYS + ["CGM"] if contract.columns.get(c) not in frame.columns]
    if missing:
        raise ValueError(f"Missing required mapped columns: {missing}")
    result = pd.DataFrame(index=frame.index)
    for canonical in CANONICAL:
        original = contract.columns.get(canonical)
        result[canonical] = frame[original] if original in frame.columns else np.nan
    for key in ("source_file", "id"):
        if result[key].isna().any() or result[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Null/empty {key} is not a valid group key")
        result[key] = result[key].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError("Null forecast timestamps are forbidden")
    if isinstance(result["date"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Timezone-aware input needs an explicit recorded-time conversion")
    result["date"] = result["date"].astype("datetime64[ns]")
    for column in NUMERIC:
        values = pd.to_numeric(result[column], errors="raise").astype(float)
        values = values.mask(values.isin(SENTINELS))
        if np.isinf(values).any():
            raise ValueError(f"Nonfinite {column}")
        if (values < 0).any():
            if column != "CGM" and contract.invalid_negative_events == "mask":
                values = values.mask(values < 0)
            else:
                raise ValueError(f"Negative {column} after sentinel masking; check source units")
        result[column] = values
    if contract.glucose_units == "mmol/L":
        result["CGM"] *= 18.0182
    if contract.basal_units == "U/hour":
        result["basal"] /= 12.0
    result["meal_label"] = result["meal_label"].map(_clean_meal_label).astype("string")
    if result.duplicated(KEYS).any():
        sample = result.loc[result.duplicated(KEYS, keep=False), KEYS].head(3).to_dict("records")
        raise ValueError(f"Duplicate source/id/timestamp keys: {sample}")
    return result.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _clean_meal_label(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-1", "-1.0", "-2", "-2.0"}:
        return np.nan
    try:
        numeric = float(text)
    except ValueError:
        return text
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError("Invalid meal_label numeric value")
    return numeric


def meal_presence(values: pd.Series) -> np.ndarray:
    """Text establishes an event, not carbohydrate grams; zero is observed absent."""

    def presence(value):
        if pd.isna(value):
            return np.nan
        try:
            return float(float(value) != 0)
        except (TypeError, ValueError):
            return 1.0

    return np.asarray([presence(v) for v in values], dtype=np.float32)


def load_frame(path: str | Path, contract: InputContract | None = None) -> pd.DataFrame:
    """Read projected Parquet/CSV columns using an explicit source contract."""
    path = Path(path)
    contract = contract or InputContract()
    if path.suffix.lower() in {".parquet", ".pq"}:
        available = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in contract.columns.values() if c in available]
        frame = pd.read_parquet(path, columns=columns)
    elif path.suffix.lower() == ".csv":
        wanted = set(contract.columns.values())
        key_types = {contract.columns[c]: "string" for c in ("id", "source_file") if c in contract.columns}
        frame = pd.read_csv(path, usecols=lambda c: c in wanted, dtype=key_types)
    else:
        raise ValueError("Input must be Parquet or CSV")
    return canonicalize(frame, contract)


def iter_projected_batches(
    path: str | Path, contract: InputContract | None = None, batch_rows: int = 200_000
) -> Iterator[pd.DataFrame]:
    """Bounded raw projected batches; group preparation validates cross-batch keys."""
    contract = contract or InputContract()
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    columns = [c for c in contract.columns.values() if c in available]
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def audit_frame(frame: pd.DataFrame) -> dict:
    """Descriptive audit only, including coverage rather than invented event zeroes."""
    clean = canonicalize(frame)
    groups = []
    for (source, participant), group in clean.groupby(["source_file", "id"], sort=False):
        minutes = group["date"].diff().dt.total_seconds().div(60)
        groups.append(
            {
                "source_file": source,
                "id": participant,
                "rows": len(group),
                "first_date": str(group.date.iloc[0]),
                "last_date": str(group.date.iloc[-1]),
                "median_cadence_minutes": float(minutes.median()) if len(group) > 1 else None,
                "five_minute_fraction": float(minutes.iloc[1:].eq(5).mean()) if len(group) > 1 else None,
                "gaps_over_15_minutes": int(minutes.gt(15).sum()),
                "coverage": {c: float(group[c].notna().mean()) for c in NUMERIC + ["meal_label"]},
            }
        )
    return {
        "rows": len(clean),
        "subjects": len(groups),
        "sources": sorted(clean.source_file.unique()),
        "sentinel_policy": "-1/-2 masked before statistics; zero events remain observed",
        "insulin_policy": "No total/split addition or reconstruction",
        "sentinel_counts": {
            c: int(pd.to_numeric(frame[c], errors="raise").isin(SENTINELS).sum()) if c in frame else 0
            for c in NUMERIC
        },
        "total_and_split_insulin_rows": int(
            (clean.insulin.notna() & (clean.basal.notna() | clean.bolus.notna())).sum()
        ),
        "zero_glucose_rows": int(clean.CGM.eq(0).sum()),
        "groups": groups,
    }
