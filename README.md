# Glucose Prediction Challenge 2026

Four-horizon glucose forecasting from past CGM and supplied future insulin and meal information.

## Results

| Evaluation | DTS-A | MARD | RMSE |
|---|---:|---:|---:|
| CarlAlbertCode, official Annual leaderboard | 90.6% | 8.1% | 20.4 mg/dL |
| Local Sandbox, public-label fit | 100.0% | 0.00% | 0.00 mg/dL |

The Annual row is the participant's displayed best metrics; the public leaderboard does not identify the exact selected prediction file for each metric. The Sandbox model fitted public test labels, so its score does not measure unseen forecasting. These rows are **not scores for the checkpoint below**.

## Public checkpoint

`model/` contains four LightGBM heads trained on the public MetaboNet TRAIN and TEST data for iLet participants. The source in `glucose/` builds the same causal feature coordinates used in training; `predict.py` runs the checkpoint from a MetaboNet raw input file and the official submission template. This is a standalone population member of the broader local Annual ensemble, not the full participant-adapted candidate.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python predict.py \
  --raw /path/to/metabonet_input.parquet \
  --template /path/to/template.parquet \
  --output predictions.parquet
```

The four heads predict 30, 60, 90 and 120 minutes. The 120-minute head uses supplied therapy through +240 minutes. Future CGM is excluded from features. On the 2,228 official Annual input origins, this package reproduced the saved checkpoint's four-head predictions exactly (maximum absolute difference 0).

Challenge data, participant-level predictions, private-data personal weights, and the full experiment archive are not redistributed. Obtain inputs and the template through [MetaboNet](https://metabo-net.org/glucose-prediction-challenge). The [official leaderboard](https://metabonetglucose-leaderboard.hf.space/?tab=annual-leaderboard-tab) records the competition result.
