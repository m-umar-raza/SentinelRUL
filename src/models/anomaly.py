import numpy as np


class AnomalyScorer:
    """Threshold anomaly detection using forecast residuals on healthy windows."""

    def __init__(self, percentile=95):
        self.percentile = percentile
        self.threshold = None

    def fit(self, residuals):
        # residuals: array of per sample mean absolute forecast errors on healthy data
        self.threshold = np.percentile(residuals, self.percentile)
        return self

    def score(self, residuals):
        return residuals

    def predict(self, residuals):
        return residuals > self.threshold
