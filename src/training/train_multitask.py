import argparse
import os
import random

import numpy as np
import torch
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
