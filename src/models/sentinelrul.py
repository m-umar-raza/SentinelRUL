import torch.nn as nn
from .backbone import GRUBackbone
from .forecast_head import ForecastHead
from .rul_head import RULHead


class SentinelRUL(nn.Module):
    """One GRU backbone shared across forecast, RUL, and anomaly heads."""

    def __init__(self, input_dim=14, hidden_dim=128, n_layers=2, dropout=0.2, horizon=5):
        super().__init__()
        self.backbone = GRUBackbone(input_dim, hidden_dim, n_layers, dropout)
        self.forecast_head = ForecastHead(hidden_dim, input_dim, horizon)
        self.rul_head = RULHead(hidden_dim, dropout)

    def forward(self, x):
        gru_out = self.backbone(x)
        encoding = gru_out[:, -1, :]
        forecast = self.forecast_head(gru_out)
        rul = self.rul_head(encoding)
        return forecast, rul
