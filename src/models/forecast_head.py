import torch.nn as nn


class ForecastHead(nn.Module):
    """Predicts next `horizon` cycles of sensor values from full GRU output."""

    def __init__(self, hidden_dim=128, output_dim=14, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(self, gru_out):
        # gru_out: (batch, seq_len, hidden_dim) — use last timestep
        h = gru_out[:, -1, :]
        out = self.fc(h)
        return out.view(-1, self.horizon, self.output_dim)
