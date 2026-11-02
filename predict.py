"""Run the public-data-trained H50 physiology checkpoint on MetaboNet inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glucose.context import _encode, weight_coordinates
from glucose.data import CANONICAL, canonicalize
from glucose.extended import FEATURE_NAMES as POST_NAMES, therapy_features
from glucose.features import GroupArray, _group_features
from glucose.relative import EXTRA_NAMES, relative_therapy_features

HORIZONS = (30, 60, 90, 120)
KEYS = ("id", "source_file", "date")
MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_models():
    schema = json.loads((MODEL_DIR / "schema.json").read_text())
    models = {}
    for horizon in HORIZONS:
        path = MODEL_DIR / f"physiology_{horizon}.txt"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != schema["model_sha256"][path.name]:
            raise ValueError(f"Checkpoint hash changed: {path.name}")
        models[horizon] = lgb.Booster(model_file=str(path))
    return schema, models


def predict(raw_path: Path, template_path: Path, output_path: Path) -> None:
    schema, models = load_models()
    template = pd.read_parquet(template_path)
    if not set(KEYS).issubset(template) or not len(template):
        raise ValueError("Template must have nonempty id/source_file/date keys")
    for horizon in HORIZONS:
        if f"pred_{horizon}" not in template:
            raise ValueError(f"Missing pred_{horizon} template column")

    keys = template[list(KEYS)].copy()
    keys["date"] = pd.to_datetime(keys["date"], errors="raise")
    if keys.isna().any().any() or keys.duplicated(list(KEYS)).any():
        raise ValueError("Missing or duplicate template origins")
    for field in ("id", "source_file"):
        keys[field] = keys[field].astype(str)

    metadata = ("weight", "height", "age", "age_of_diagnosis")
    available = set(pq.ParquetFile(raw_path).schema_arrow.names)
    required = set(CANONICAL) - {"meal_label"}
    if not required.issubset(available):
        raise ValueError(f"Raw file lacks {sorted(required - available)}")
    columns = [name for name in (*CANONICAL, *metadata) if name in available]
    raw = pd.read_parquet(raw_path, columns=columns)
    for name in metadata:
        if name not in raw:
            raw[name] = np.nan
    for field in ("id", "source_file"):
        raw[field] = raw[field].astype(str)

    raw_groups = raw.groupby(["source_file", "id"], sort=False)
    origins = keys.groupby(["source_file", "id"], sort=False).indices
    missing = set(origins) - set(raw_groups.indices)
    if missing:
        raise ValueError(f"Template contains {len(missing)} absent participant streams")

    result = np.full((len(keys), 4), np.nan, dtype=np.float64)
    for number, (identity, positions) in enumerate(origins.items(), start=1):
        parent = raw_groups.get_group(identity)
        frame = canonicalize(parent)
        query = keys.iloc[positions]["date"]
        timestamps = query.to_numpy("datetime64[ns]").astype(np.int64)
        base, _, base_names, masks = _group_features(
            GroupArray.from_frame(frame), timestamps, 120.0, True
        )
        relative = relative_therapy_features(base, base_names, masks)
        post = therapy_features(frame, timestamps)
        full = np.column_stack((base, relative, post)).astype(np.float32)
        full_names = [*base_names, *EXTRA_NAMES, *POST_NAMES]
        context = _encode(parent, query)
        if full.shape[1] != 773 or context.shape != (len(positions), 10):
            raise ValueError("Checkpoint feature width changed")

        for head, horizon in enumerate(HORIZONS):
            width = 495 if horizon < 120 else 773
            names = base_names if horizon < 120 else full_names
            if names != schema["model_feature_names"][str(horizon)]:
                raise ValueError(f"Checkpoint schema differs at {horizon} minutes")
            if list(schema["context_feature_names"]) != [
                "weight", "weight_valid", "height", "height_valid",
                "age", "age_valid", "age_of_diagnosis", "age_of_diagnosis_valid",
                "diabetes_duration_years", "diabetes_duration_valid",
            ]:
                raise ValueError("Context schema changed")
            x = full[:, :width]
            weight, _ = weight_coordinates(x, names, context, horizon)
            features = np.column_stack((x, context, weight)).astype(np.float32)
            if features.shape[1] != models[horizon].num_feature():
                raise ValueError("Model input width changed")
            result[positions, head] = np.maximum(
                0.0, models[horizon].predict(features, num_threads=4)
            )
        print(f"Processed {number}/{len(origins)} participant streams", flush=True)

    if not np.isfinite(result).all():
        raise ValueError("Some predictions are missing or nonfinite")
    for head, horizon in enumerate(HORIZONS):
        template[f"pred_{horizon}"] = result[:, head]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_parquet(output_path, index=False)
    print(f"Wrote {len(template)} four-horizon predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.raw, args.template, args.output)


if __name__ == "__main__":
    main()
