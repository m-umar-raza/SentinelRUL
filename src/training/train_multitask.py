import argparse
import os
import math
import random

import numpy as np
import torch
from torch.optim import Adam
import yaml

from ..data.loader import load_train
from ..data.preprocess import prepare_train, SENSOR_COLS
from ..data.windows import make_windows
from ..data.dataset import CMAPSSWindows
from ..models import SentinelRUL
from .losses import JointLoss


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


def load_pretrained_backbone(model, checkpoint_path):
    """Copies the forecast pretrained GRU weights into the multitask model's backbone."""
    if not os.path.exists(checkpoint_path):
        print(f"no pretrained backbone at {checkpoint_path}, training backbone from scratch")
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state"]
    backbone_state = {k: v for k, v in state.items() if k.startswith("backbone.")}
    model.load_state_dict(backbone_state, strict=False)
    print(f"loaded pretrained backbone from {checkpoint_path} ({len(backbone_state)} tensors)")


class MultitaskTrainer:
    """Joint training loop that starts from a forecast pretrained backbone."""

    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-3,
        weight_decay=1e-5,
        alpha=0.5,
        freeze_backbone_epochs=0,
        checkpoint_dir="checkpoints/multitask",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = JointLoss(forecast_weight=alpha, rul_weight=1.0)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_rmse = math.inf

    def _set_backbone_trainable(self, trainable):
        for p in self.model.backbone.parameters():
            p.requires_grad = trainable

    def train_epoch(self, loader, epoch):
        self._set_backbone_trainable(epoch > self.freeze_backbone_epochs)
        self.model.train()
        running = {"total": 0.0, "forecast": 0.0, "rul": 0.0, "n": 0}
        for x, y_rul, y_fore in loader:
            x = x.to(self.device)
            y_rul = y_rul.to(self.device)
            y_fore = y_fore.to(self.device)
            self.optimizer.zero_grad()
            forecast_pred, rul_pred = self.model(x)
            loss, parts = self.criterion(forecast_pred, rul_pred, y_fore, y_rul)
            loss.backward()
            self.optimizer.step()
            bs = x.size(0)
            running["total"] += loss.item() * bs
            running["forecast"] += parts["forecast"].item() * bs
            running["rul"] += parts["rul"].item() * bs
            running["n"] += bs
        n = running["n"]
        return {k: running[k] / n for k in ("total", "forecast", "rul")}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        sse_rul = 0.0
        sse_fore = 0.0
        n_rul = 0
        n_fore = 0
        for x, y_rul, y_fore in loader:
            x = x.to(self.device)
            y_rul = y_rul.to(self.device)
            y_fore = y_fore.to(self.device)
            forecast_pred, rul_pred = self.model(x)
            sse_rul += ((rul_pred - y_rul) ** 2).sum().item()
            n_rul += y_rul.numel()
            sse_fore += ((forecast_pred - y_fore) ** 2).sum().item()
            n_fore += y_fore.numel()
        return {
            "val_rul_rmse": math.sqrt(sse_rul / n_rul),
            "val_forecast_rmse": math.sqrt(sse_fore / n_fore),
        }

    def fit(self, train_loader, val_loader, epochs, early_stop_patience=None):
        history = []
        epochs_since_improvement = 0
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            row = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(row)
            print(
                f"epoch {epoch:3d} | "
                f"train_loss {train_metrics['total']:.4f} "
                f"(fore {train_metrics['forecast']:.4f}, rul {train_metrics['rul']:.4f}) | "
                f"val_rul_rmse {val_metrics['val_rul_rmse']:.3f} "
                f"val_fore_rmse {val_metrics['val_forecast_rmse']:.4f}"
            )
            if val_metrics["val_rul_rmse"] < self.best_val_rmse:
                self.best_val_rmse = val_metrics["val_rul_rmse"]
                epochs_since_improvement = 0
                self.save("best.pt")
            else:
                epochs_since_improvement += 1
                if early_stop_patience and epochs_since_improvement >= early_stop_patience:
                    print(f"no improvement in {early_stop_patience} epochs, stopping early")
                    break
        self.save("last.pt")
        return history

    def save(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({"model_state": self.model.state_dict()}, path)
