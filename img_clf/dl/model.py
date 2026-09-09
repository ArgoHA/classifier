"""The timm model factory, shared by training and by everything that reloads a checkpoint.

Lives apart from train.py so that helpers (utils.auto_batch_size) and the export / vis
entrypoints can build a model without importing the training entrypoint - and without
train.py importing them back, which is a circular import.
"""

from pathlib import Path
from typing import List

import timm
from loguru import logger
from timm.utils.model import freeze
from torch import nn

from img_clf.dl.ckpt import load_and_describe


def build_model(
    model_name: str, pretrained: bool, num_labels: int, device: str, layers_to_train: int
) -> nn.Module:
    """`layers_to_train`: -1 trains everything, N >= 0 trains the last N parameter *groups*
    and freezes the rest, so 0 freezes the whole network.

    Groups are the model's top-level children in forward order (`_parameter_groups`), so
    "N" means a comparable amount of network across architectures. The previous version
    sliced `list(model.parameters())[:-N]`, i.e. it counted individual tensors in
    registration order - "N" meant a different amount of the network for every
    architecture, and could freeze half of a block while leaving its norm trainable.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_labels)
    if layers_to_train == -1:
        return model.to(device)
    if layers_to_train < 0:
        raise ValueError(
            f"layers_to_train must be -1 (train everything) or >= 0, got {layers_to_train}"
        )

    groups = _parameter_groups(model)
    # `len(groups) - N`, not `-N`: `groups[:-0]` is the *empty* slice, so N=0 would freeze
    # nothing when it means freeze everything, and N=-2 would freeze the first two groups.
    to_freeze = groups[: max(len(groups) - layers_to_train, 0)]
    if not to_freeze:
        logger.warning(
            f"layers_to_train={layers_to_train} covers all {len(groups)} groups of "
            f"{model_name}; nothing frozen"
        )
    else:
        # FrozenBatchNorm
        freeze(model, to_freeze, include_bn_running_stats=False)
        model._frozen_groups = to_freeze
        logger.info(f"Frozen {len(to_freeze)}/{len(groups)} groups: {to_freeze}")
    return model.to(device)


def _parameter_groups(model: nn.Module) -> List[str]:
    """Top-level module names, ordered as the forward pass visits them."""
    return [name for name, _ in model.named_children()]


def head_param_names(model: nn.Module) -> set:
    """Names of the classifier's own parameters, asked of timm rather than matched by name.

    `"head" in name` also catches efficientnet's `conv_head` - a 410k-parameter backbone
    convolution, the largest conv in the network - which then trains at the head learning
    rate and silently defeats `backbone_lr` for the default model.
    """
    get_classifier = getattr(model, "get_classifier", None)
    head = get_classifier() if callable(get_classifier) else None
    if head is None:
        return set()
    head_params = {id(p) for p in head.parameters()}
    return {name for name, p in model.named_parameters() if id(p) in head_params}


def prepare_model(model_name: str, model_path: Path, num_labels: int, device: str) -> nn.Module:
    model = build_model(
        model_name=model_name,
        num_labels=num_labels,
        pretrained=False,
        device=device,
        layers_to_train=-1,
    )
    checkpoint, _ = load_and_describe(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
