from .mlp import ResidualMLP
from .temporal_cnn import ResidualCNN
from .transformer import ResidualTransformer

MODELS = {"mlp": ResidualMLP, "transformer": ResidualTransformer, "temporal_cnn": ResidualCNN}


def build_model(name: str, lookback: int, **params):
    return MODELS[name](lookback=lookback, **params)
