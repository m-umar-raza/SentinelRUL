import torch.nn as nn


class RULHead(nn.Module):
    """Regresses scalar RUL from the last GRU hidden state."""

    def __init__(self, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, encoding):
        # encoding: (batch, hidden_dim) — output of backbone.encode()
        return self.net(encoding).squeeze(-1)
