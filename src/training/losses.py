import torch
import torch.nn as nn


class JointLoss(nn.Module):
    """Weighted sum of forecast MSE and RUL MSE."""

    def __init__(self, forecast_weight=0.5, rul_weight=1.0):
        super().__init__()
        self.forecast_weight = forecast_weight
        self.rul_weight = rul_weight
        self.mse = nn.MSELoss()

    def forward(self, forecast_pred, rul_pred, y_fore, y_rul):
        forecast_loss = self.mse(forecast_pred, y_fore)
        rul_loss = self.mse(rul_pred, y_rul)
        total = self.forecast_weight * forecast_loss + self.rul_weight * rul_loss
        return total, {"forecast": forecast_loss.detach(), "rul": rul_loss.detach()}
