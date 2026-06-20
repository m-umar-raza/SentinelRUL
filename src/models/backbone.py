import torch.nn as nn


class GRUBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        return out  # (batch, seq_len, hidden_dim)

    def encode(self, x):
        out = self.forward(x)
        return out[:, -1, :]  # last timestep for regression heads
