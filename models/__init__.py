import logging
import pickle

import torch
from torch import nn

from .slow_norm import SlowNorm, graphnorm
from .graph_cross_attention import GraphCrossAttention
from .action_model import ActionModel
from .mcts_model import MCTS_like_model
from .action_net import ActionNet
from .action_embedding_model import ActionEmbeddingModel
from .bundle_net import BundleNet, DummyActionModel, TreeNet
from .categorical_masked import CategoricalMasked

__all__ = [
    "SlowNorm",
    "graphnorm",
    "GraphCrossAttention",
    "ActionModel",
    "MCTS_like_model",
    "ActionNet",
    "ActionEmbeddingModel",
    "BundleNet",
    "DummyActionModel",
    "TreeNet",
    "CategoricalMasked",
    "load_agent_checkpoint",
]


def load_agent_checkpoint(model_path, build_agent=None):
    """Load an agent from ``model_path`` with graceful fallback.

    Two checkpoint styles exist in this repo: whole pickled / torch-saved
    ``nn.Module``, and a plain ``state_dict`` that must be rebuilt through a
    config-driven factory. Loading order: whole pickled model first; if
    unavailable, fall back to ``torch.load`` + ``build_agent()`` +
    ``load_state_dict``.

    Args:
        model_path: path to a checkpoint file.
        build_agent: zero-arg factory returning an uninitialized ``nn.Module``.
            Required only when the checkpoint turns out to be a state_dict.

    Returns:
        The loaded ``nn.Module``.
    """
    # 1) try a plain pickle of the whole model
    try:
        with open(model_path, "rb") as reader:
            obj = pickle.load(reader)
        if isinstance(obj, nn.Module):
            logging.info("loaded whole pickled agent from %s", model_path)
            return obj
    except Exception:
        # not a plain pickle (e.g. torch's zip checkpoint format) — fall through
        pass

    # 2) torch.load: whole model or state_dict.
    # weights_only=False is needed for torch-saved whole models; only load
    # checkpoints you trust.
    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        logging.info("loaded torch-saved agent from %s", model_path)
        return obj

    # 3) state_dict: rebuild from the caller's config
    if build_agent is None:
        raise ValueError(
            f"{model_path} looks like a state_dict; "
            "pass build_agent to reconstruct the model architecture"
        )
    logging.info("rebuilding agent from state_dict %s", model_path)
    agent = build_agent()
    agent.load_state_dict(obj)
    return agent
