import argparse
import os
import math
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import yaml

from ..models.backbone import GRUBackbone
from ..models.forecast_head import ForecastHead
from ..data.loader import load_train
from ..data.preprocess import prepare_train, SENSOR_COLS
from ..data.windows import make_windows
from ..data.dataset import CMAPSSWindows


class ForecastOnly(nn.Module):
    """Backbone and forecast head only, used to pretrain the shared GRU encoder."""

    def __init__(self, input_dim=14, hidden_dim=128, n_layers=2, dropout=0.2, horizon=5):
        super().__init__()
        self.backbone = GRUBackbone(input_dim, hidden_dim, n_layers, dropout)
        self.forecast_head = ForecastHead(hidden_dim, input_dim, horizon)

    def forward(self, x):
        gru_out = self.backbone(x)
        return self.forecast_head(gru_out)


class ForecastTrainer:
    """Training loop for forecast only pretraining. Minimises MSE on next N sensor cycles."""

    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=1.0,
        checkpoint_dir="checkpoints/forecast",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_rmse = math.inf

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        n = 0
        for x, _, y_fore in loader:
            x, y_fore = x.to(self.device), y_fore.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y_fore)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
        return total_loss / n

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        sse = 0.0
        n = 0
        for x, _, y_fore in loader:
            x, y_fore = x.to(self.device), y_fore.to(self.device)
            pred = self.model(x)
            sse += ((pred - y_fore) ** 2).sum().item()
            n += y_fore.numel()
        return math.sqrt(sse / n)

    def fit(self, train_loader, val_loader, epochs):
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_rmse = self.validate(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
            print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_rmse {val_rmse:.4f}")
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self._save("best.pt")
        self._save("last.pt")
        return history

    def _save(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({"model_state": self.model.state_dict()}, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_engine(df, val_engines, seed):
    rng = np.random.default_rng(seed)
    engines = df["engine_id"].unique()
    rng.shuffle(engines)
    val_ids = set(engines[:val_engines])
    return df[~df["engine_id"].isin(val_ids)], df[df["engine_id"].isin(val_ids)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "forecast_config.yaml"),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    raw = load_train()
    full, mins, maxs = prepare_train(raw, rul_clip=cfg["data"]["rul_clip"])
    train_df, val_df = split_by_engine(full, cfg["data"]["val_engines"], cfg["training"]["seed"])

    window = cfg["data"]["window_size"]
    horizon = cfg["data"]["forecast_horizon"]
    X_tr, y_rul_tr, y_fore_tr = make_windows(train_df, window, horizon)
    X_val, y_rul_val, y_fore_val = make_windows(val_df, window, horizon)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        CMAPSSWindows(X_tr, y_rul_tr, y_fore_tr), batch_size=bs, shuffle=True
    )
    val_loader = DataLoader(
        CMAPSSWindows(X_val, y_rul_val, y_fore_val), batch_size=bs, shuffle=False
    )

    model = ForecastOnly(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        horizon=cfg["model"]["horizon"],
    )

    trainer = ForecastTrainer(
        model,
        device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )

    history = trainer.fit(train_loader, val_loader, cfg["training"]["epochs"])

    ckpt_dir = cfg["output"]["checkpoint_dir"]
    scaler = {"mins": mins.to_dict(), "maxs": maxs.to_dict(), "sensors": SENSOR_COLS}
    with open(os.path.join(ckpt_dir, "scaler.json"), "w") as f:
        json.dump(scaler, f)
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"best val_forecast_rmse: {trainer.best_val_rmse:.4f}")


if __name__ == "__main__":
    main()
