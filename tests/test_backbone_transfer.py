import torch

from src.models import SentinelRUL
from src.training.train_forecast import ForecastOnly
from src.training.train_multitask import load_pretrained_backbone


def test_load_pretrained_backbone_copies_gru_weights(tmp_path):
    forecast_model = ForecastOnly(input_dim=14, hidden_dim=8, n_layers=1, horizon=5)
    ckpt_path = tmp_path / "forecast_best.pt"
    torch.save({"model_state": forecast_model.state_dict()}, ckpt_path)

    multitask_model = SentinelRUL(input_dim=14, hidden_dim=8, n_layers=1, horizon=5)
    load_pretrained_backbone(multitask_model, str(ckpt_path))

    for name, param in multitask_model.backbone.named_parameters():
        expected = forecast_model.backbone.state_dict()[name]
        assert torch.equal(param, expected)


def test_load_pretrained_backbone_missing_file_is_noop(tmp_path):
    model = SentinelRUL(input_dim=14, hidden_dim=8, n_layers=1, horizon=5)
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}

    load_pretrained_backbone(model, str(tmp_path / "does_not_exist.pt"))

    for name, param in model.backbone.named_parameters():
        assert torch.equal(param, before[name])
