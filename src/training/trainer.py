import os
import math
import torch
from torch.optim import Adam

from .losses import JointLoss


class Trainer:
    """Joint training loop for SentinelRUL. Tracks val RUL RMSE and saves best checkpoint."""

    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-3,
        weight_decay=1e-5,
        forecast_weight=0.5,
        rul_weight=1.0,
        checkpoint_dir="checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = JointLoss(forecast_weight, rul_weight)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_rmse = math.inf

    def _step(self, batch):
        x, y_rul, y_fore = [t.to(self.device) for t in batch]
        forecast_pred, rul_pred = self.model(x)
        loss, parts = self.criterion(forecast_pred, rul_pred, y_fore, y_rul)
        return loss, parts, rul_pred, y_rul

    def train_epoch(self, loader):
        self.model.train()
        running = {"total": 0.0, "forecast": 0.0, "rul": 0.0, "n": 0}
        for batch in loader:
            self.optimizer.zero_grad()
            loss, parts, _, _ = self._step(batch)
            loss.backward()
            self.optimizer.step()
            bs = batch[0].size(0)
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
        for batch in loader:
            _, _, rul_pred, y_rul = self._step(batch)
            x, _, y_fore = [t.to(self.device) for t in batch]
            forecast_pred, _ = self.model(x)
            sse_rul += ((rul_pred - y_rul) ** 2).sum().item()
            n_rul += y_rul.numel()
            sse_fore += ((forecast_pred - y_fore) ** 2).sum().item()
            n_fore += y_fore.numel()
        return {
            "val_rul_rmse": math.sqrt(sse_rul / n_rul),
            "val_forecast_rmse": math.sqrt(sse_fore / n_fore),
        }

    def fit(self, train_loader, val_loader, epochs):
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
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
                self.save("best.pt")
        self.save("last.pt")
        return history

    def save(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({"model_state": self.model.state_dict()}, path)
