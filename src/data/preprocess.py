import numpy as np
import pandas as pd

# sensors near constant in FD001, no useful signal
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SENSOR_COLS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP_SENSORS]


def add_rul_labels(df, rul_clip=125):
    """Piecewise linear RUL, flat at rul_clip until degradation kicks in."""
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_clip)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def fit_scaler(train_df):
    """Per column min and max fitted on training data."""
    mins = train_df[SENSOR_COLS].min()
    maxs = train_df[SENSOR_COLS].max()
    return mins, maxs


def normalize(df, mins, maxs):
    df = df.copy()
    denom = (maxs - mins).replace(0, 1)
    df[SENSOR_COLS] = (df[SENSOR_COLS] - mins) / denom
    return df


def prepare_train(train_df, rul_clip=125):
    df = add_rul_labels(train_df.copy(), rul_clip)
    mins, maxs = fit_scaler(df)
    df = normalize(df, mins, maxs)
    return df, mins, maxs


def prepare_test(test_df, rul_df, mins, maxs):
    df = test_df.copy()
    # for test, true RUL is last cycle RUL from rul_df plus cycles remaining
    # we keep the last window per engine for evaluation
    df = normalize(df, mins, maxs)
    return df
