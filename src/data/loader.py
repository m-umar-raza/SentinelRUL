import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

COLS = ["engine_id", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


def load_train():
    return _load("train_FD001.txt")


def load_test():
    return _load("test_FD001.txt")


def load_rul():
    path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    return pd.read_csv(path, header=None, names=["rul"])


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df
