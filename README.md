# SentinelRUL

Predictive maintenance system built on the NASA C-MAPSS turbofan dataset. Uses a shared GRU backbone to jointly handle three tasks: sensor forecasting, remaining useful life (RUL) regression, and anomaly detection, all from a single encoded window.

## Problem

Turbofan engines degrade over time. Unplanned failures are expensive. The goal is to predict how many cycles an engine has left (RUL) and flag anomalous behavior before it becomes critical, using only raw sensor readings.

## Approach

One GRU encodes a sliding window of sensor data. Three heads read from that shared representation:

* Forecast head: predicts the next 5 cycles of sensor values
* RUL head: regresses remaining useful life directly
* Anomaly layer: computes forecast residuals, flags sequences that exceed the healthy baseline error distribution

This means anomaly detection comes for free once the forecaster is trained, no separate anomaly model needed.

## Dataset

NASA C-MAPSS FD001 subset:

* 100 training engines, 100 test engines
* Single operating condition, single fault mode
* 21 sensors, 3 operational settings
* RUL labels provided for test set

## Architecture

```
sensor window (30 cycles x 14 sensors)
        |
   GRU Backbone
   hidden_dim=128, n_layers=2
        |
   shared representation
        |
   forecast head   rul head   anomaly score
```

## Results

| Capability | Metric | Score |
|---|---|---|
| Forecasting | RMSE (5 step) | TBD |
| RUL | RMSE | TBD |
| RUL | NASA Score | TBD |
| Anomaly | Precision / Recall | TBD |
| Anomaly | Avg lead time (cycles) | TBD |

## Demo

Coming soon. FastAPI service plus dashboard.

## Setup

```bash
git clone https://github.com/m-umar-raza/SentinelRUL.git
cd SentinelRUL
pip install -r requirements.txt
python data/download.py
```

## Training

Training happens in two stages so the shared backbone learns sensor dynamics before it ever sees an RUL label.

1. **Forecast pretraining** (backbone only, self supervised on the next 5 sensor cycles):
   ```bash
   python -m src.training.train_forecast --config src/training/configs/forecast_config.yaml
   ```
   Best checkpoint lands at `checkpoints/forecast/best.pt`.

2. **Multitask training** (RUL head plus forecast head, backbone initialised from step 1):
   ```bash
   python -m src.training.train_multitask --config src/training/configs/multitask_config.yaml
   ```
   Loads `checkpoints/forecast/best.pt` if present, otherwise starts the backbone from scratch. Joint loss is `alpha * forecast_mse + rul_mse` with `alpha=0.5`. Best checkpoint lands at `checkpoints/multitask/best.pt`.

Both stages were trained on Kaggle GPUs using the notebooks under `notebooks/`, since this repo is developed on a CPU only machine.
