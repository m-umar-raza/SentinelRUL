import torch
from torch.utils.data import Dataset


class CMAPSSWindows(Dataset):
    def __init__(self, X, y_rul, y_fore):
        self.X = torch.from_numpy(X)
        self.y_rul = torch.from_numpy(y_rul)
        self.y_fore = torch.from_numpy(y_fore)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_rul[idx], self.y_fore[idx]
