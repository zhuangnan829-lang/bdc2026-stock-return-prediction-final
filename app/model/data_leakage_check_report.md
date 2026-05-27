# Data Leakage Check Report

- Generated at: 2026-05-24 21:42:25
- Overall status: **PASS**

| Check | Status | Detail |
|---|---:|---|
| required file: train_features | PASS | app/temp/train_features.csv exists |
| required file: predict_features | PASS | app/temp/predict_features.csv exists |
| required file: model_meta | PASS | app/model/model_meta.json exists |
| required file: walk_forward_metrics | PASS | app/model/walk_forward_metrics.csv exists |
| required file: walk_forward_predictions | PASS | app/model/walk_forward_predictions.csv exists |
| feature date <= prediction date | PASS | 1500 prediction rows checked in app/temp/predict_features.csv |
| training label horizon is forward only | PASS | 155090 labeled rows have strictly later 5-trading-day label horizons |
| label fields excluded from model features | PASS | 20 model feature columns checked; no target/future/prediction label fields included |
| walk-forward train/validation order | PASS | 3 folds checked; all train windows end before validation windows |
| walk-forward windows reproducible from feature dates | PASS | Rebuilt 3 folds from train feature dates and matched model_meta.json |
