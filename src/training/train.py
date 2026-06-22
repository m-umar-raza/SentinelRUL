import argparse
import os
import random
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.loader import load_train
from ..data.preprocess import prepare_train, SENSOR_COLS
from ..data.windows import make_windows
from ..data.dataset import CMAPSSWindows
from ..models import SentinelRUL
from .trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_engine(df, val_engines, seed):
    rng = np.random.default_rng(seed)
    engines = df["engine_id"].unique()
    rng.shuffle(engines)
    val_ids = set(engines[:val_engines])
    train_df = df[~df["engine_id"].isin(val_ids)]
    val_df = df[df["engine_id"].isin(val_ids)]
    return train_df, val_df


def build_loaders(cfg):
    raw = load_train()
    full, mins, maxs = prepare_train(raw, rul_clip=cfg["data"]["rul_clip"])

    train_df, val_df = split_by_engine(
        full, cfg["data"]["val_engines"], cfg["training"]["seed"]
    )

    window = cfg["data"]["window_size"]
    horizon = cfg["data"]["forecast_horizon"]
    X_tr, y_rul_tr, y_fore_tr = make_windows(train_df, window, horizon)
    X_val, y_rul_val, y_fore_val = make_windows(val_df, window, horizon)

    train_ds = CMAPSSWindows(X_tr, y_rul_tr, y_fore_tr)
    val_ds = CMAPSSWindows(X_val, y_rul_val, y_fore_val)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    scaler = {"mins": mins.to_dict(), "maxs": maxs.to_dict(), "sensors": SENSOR_COLS}
    return train_loader, val_loader, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, scaler = build_loaders(cfg)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = SentinelRUL(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        horizon=cfg["model"]["horizon"],
    )

    trainer = Trainer(
        model,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        forecast_weight=cfg["training"]["forecast_weight"],
        rul_weight=cfg["training"]["rul_weight"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    history = trainer.fit(train_loader, val_loader, cfg["training"]["epochs"])

    with open(os.path.join(cfg["output"]["checkpoint_dir"], "scaler.json"), "w") as f:
        json.dump(scaler, f)
    with open(os.path.join(cfg["output"]["checkpoint_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"best val_rul_rmse: {trainer.best_val_rmse:.3f}")


if __name__ == "__main__":
    main()
