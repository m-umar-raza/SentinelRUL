import numpy as np
from .preprocess import SENSOR_COLS

WINDOW_SIZE = 30
FORECAST_HORIZON = 5


def make_windows(df, window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON):
    """
    Returns:
      X      (N, window_size, n_sensors)
      y_rul  (N,)
      y_fore (N, forecast_horizon, n_sensors)
    """
    X, y_rul, y_fore = [], [], []

    for engine_id, group in df.groupby("engine_id"):
        sensors = group[SENSOR_COLS].values
        rul = group["rul"].values
        n = len(sensors)

        for i in range(n - window_size - forecast_horizon + 1):
            X.append(sensors[i : i + window_size])
            y_rul.append(rul[i + window_size - 1])
            y_fore.append(sensors[i + window_size : i + window_size + forecast_horizon])

    return (
        np.array(X, dtype=np.float32),
        np.array(y_rul, dtype=np.float32),
        np.array(y_fore, dtype=np.float32),
    )


def make_test_windows(df, window_size=WINDOW_SIZE):
    """Returns last window per engine for inference."""
    X = []
    engine_ids = []

    for engine_id, group in df.groupby("engine_id"):
        sensors = group[SENSOR_COLS].values
        if len(sensors) >= window_size:
            X.append(sensors[len(sensors) - window_size:])
        else:
            pad = np.zeros((window_size - len(sensors), sensors.shape[1]), dtype=np.float32)
            X.append(np.vstack([pad, sensors]))
        engine_ids.append(engine_id)

    return np.array(X, dtype=np.float32), engine_ids
