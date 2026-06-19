import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

COLS = ["engine_id", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(filename):
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def main():
    train = load_raw("train_FD001.txt")
    test = load_raw("test_FD001.txt")
    rul = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None, names=["rul"])

    print("=== train_FD001 ===")
    print(f"  shape: {train.shape}")
    print(f"  engines: {train['engine_id'].nunique()}")
    print(f"  total cycles: {len(train)}")

    cycle_lengths = train.groupby("engine_id")["cycle"].max()
    print(f"\n  cycle length per engine:")
    print(f"    min={cycle_lengths.min()}, max={cycle_lengths.max()}, "
          f"mean={cycle_lengths.mean():.1f}, median={cycle_lengths.median():.0f}")

    print("\n=== test_FD001 ===")
    print(f"  shape: {test.shape}")
    print(f"  engines: {test['engine_id'].nunique()}")

    print("\n=== RUL labels ===")
    print(f"  shape: {rul.shape}")
    print(f"  min={rul['rul'].min()}, max={rul['rul'].max()}, mean={rul['rul'].mean():.1f}")

    print("\n=== sensor stats (train) ===")
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    stats = train[sensor_cols].agg(["mean", "std", "min", "max"])
    # flag near-constant sensors (std < 0.01)
    low_var = [c for c in sensor_cols if train[c].std() < 0.01]
    print(f"  near-constant sensors (std < 0.01): {low_var}")
    print(stats.T.to_string())


if __name__ == "__main__":
    main()
