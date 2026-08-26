"""Checkpoint envelope: weights plus the facts needed to run them.

A bare `state_dict()` says nothing about how to *use* the weights, so every consumer had
to be told `model_name`, `n_outputs`, `input_size` and the normalization separately. Training
writes `{"model": sd, "meta": {...}}`, and `load_and_describe` recovers the rest.

`unwrap_checkpoint` passes a bare `state_dict` straight through, so checkpoints written
before this envelope existed keep loading.

Resolution order for a fact: the checkpoint's own `meta` (travels with the file), then the
`config.yaml` training freezes beside it (stays behind in the run dir), then an explicit
constructor argument's default. Never a silent global.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import yaml
from loguru import logger
import timm

CKPT_VERSION = 1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_norm(model_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """(mean, std) the pretrained weights of `model_name` were trained with."""
    try:
        cfg = timm.get_pretrained_cfg(model_name)
        mean, std = tuple(cfg.mean), tuple(cfg.std)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(f"unexpected mean/std arity: {mean}, {std}")
        return mean, std
    except Exception as e:
        logger.warning(f"could not resolve norm for {model_name!r} ({e}); using ImageNet stats")
        return IMAGENET_MEAN, IMAGENET_STD


def unwrap_checkpoint(state: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Split any checkpoint format into (state_dict, meta).

    `save_checkpoint` writes {"model": sd, "meta": {...}}; anything trained before that is
    a bare state_dict and yields an empty meta.
    """
    if "model" in state and isinstance(state["model"], dict):
        return state["model"], state.get("meta") or {}
    return state, {}


# Exact types, not isinstance: np.float64 subclasses float but still pickles as a numpy
# scalar, which weights_only rejects.
_PLAIN_TYPES = (str, int, float, bool)


def _assert_plain(value: Any, path: str) -> None:
    if value is None or type(value) in _PLAIN_TYPES:
        return
    if type(value) is dict:
        for key, item in value.items():
            _assert_plain(key, f"{path}.{key}")
            _assert_plain(item, f"{path}.{key}")
        return
    if type(value) in (list, tuple):
        for i, item in enumerate(value):
            _assert_plain(item, f"{path}[{i}]")
        return
    raise TypeError(
        f"checkpoint meta must be plain python, but {path} is {type(value).__name__}. "
        "Every loader here passes weights_only=True, which rejects numpy scalars and "
        "OmegaConf nodes - cast it (int/float/str/list/dict) in ckpt_meta first."
    )


def save_checkpoint(
    path: Union[str, Path], state_dict: Dict[str, torch.Tensor], meta: Dict[str, Any]
) -> None:
    """Write the envelope `unwrap_checkpoint` reads.

    Exists for two reasons, both about failing loudly:

    - The envelope keys live here, next to the reader. Inlining `torch.save({"model":
      ...})` at each call site risks a typo, and a typo is *silent*: `unwrap_checkpoint`
      tolerates a bare state_dict for back-compat, so a misspelled key loads fine with an
      empty meta and the wrappers quietly fall back to ImageNet normalization.
    - `meta` is validated here. A numpy scalar or OmegaConf node saves without complaint
      and then fails inside `torch.load(weights_only=True)` - in whatever service loads
      the checkpoint, pointing at the loader rather than at the line that wrote it.
    """
    _assert_plain(meta, "meta")
    torch.save({"model": state_dict, "meta": meta}, path)


def ckpt_meta(cfg, mean: Sequence[float], std: Sequence[float]) -> Dict[str, Any]:
    """What a checkpoint cannot otherwise carry once it leaves its run directory."""
    return {
        "ckpt_version": CKPT_VERSION,
        "model_name": str(cfg.model_name),
        "num_classes": len(cfg.train.label_to_name),
        "label_to_name": {int(k): str(v) for k, v in cfg.train.label_to_name.items()},
        "img_size": [int(v) for v in cfg.train.img_size],
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
    }


def sibling_config(ckpt: Path) -> Dict[str, Any]:
    """`config.yaml` that training freezes next to the checkpoint (dl/train.py)."""
    p = ckpt.parent / "config.yaml"
    if not p.is_file():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:  # a malformed sidecar must not block loading the weights
        logger.warning(f"ignoring unreadable {p}: {e}")
        return {}


def describe(sd: Dict[str, torch.Tensor], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Architecture facts recoverable from the weights themselves.

    Only `num_classes` is truly recoverable - it is the classifier head's output dim, and
    timm names that layer differently per family, so we take the last 2-D weight in the
    dict. Everything else (model_name, img_size, normalization) is preprocessing or
    architecture choice that no tensor records.
    """
    meta = meta or {}
    num_classes = meta.get("num_classes")
    head_out = None
    for tensor in reversed(list(sd.values())):
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            head_out = int(tensor.shape[0])
            break
    if head_out is None:  # e.g. a head that is a 1x1 conv
        for tensor in reversed(list(sd.values())):
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 1:
                head_out = int(tensor.shape[0])
                break
    if num_classes is None:
        num_classes = head_out
    elif head_out is not None and head_out != num_classes:
        logger.warning(
            f"checkpoint meta says num_classes={num_classes} but the head has {head_out} "
            "outputs; trusting the weights"
        )
        num_classes = head_out

    return {
        "model_name": meta.get("model_name"),
        "num_classes": num_classes,
        "label_to_name": _coerce_names(meta.get("label_to_name")),
        "img_size": _coerce_size(meta.get("img_size")),
        "mean": _coerce_norm(meta.get("mean")),
        "std": _coerce_norm(meta.get("std")),
    }


def load_and_describe(
    path: Union[str, Path], map_location: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """-> (state_dict, info). Reads the file once; callers reuse the state_dict."""
    p = Path(path)
    sd, meta = unwrap_checkpoint(torch.load(p, map_location=map_location, weights_only=True))
    info = describe(sd, meta)

    # Fall back to the frozen sidecar config for anything meta did not carry - that is the
    # path a pre-envelope checkpoint takes, and it is why old run dirs stay usable.
    cfg = sibling_config(p)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    if info["model_name"] is None:
        info["model_name"] = cfg.get("model_name")
    if info["label_to_name"] is None:
        info["label_to_name"] = _coerce_names(train_cfg.get("label_to_name"))
    if info["img_size"] is None:
        info["img_size"] = _coerce_size(train_cfg.get("img_size"))

    # Normalization is not in the old config schema at all; derive it from the model name,
    # which is exactly what training did implicitly.
    if info["mean"] is None or info["std"] is None:
        if info["model_name"]:
            info["mean"], info["std"] = resolve_norm(info["model_name"])
        else:
            logger.warning(f"{p.name}: no model_name to resolve normalization; using ImageNet")
            info["mean"], info["std"] = IMAGENET_MEAN, IMAGENET_STD

    if info["num_classes"] is None and info["label_to_name"]:
        info["num_classes"] = len(info["label_to_name"])
    return sd, info


def inspect(path: Union[str, Path]) -> Dict[str, Any]:
    """-> {model_name, num_classes, label_to_name, img_size, mean, std}."""
    return load_and_describe(path)[1]


def _coerce_names(raw: Any) -> Optional[Dict[int, str]]:
    if not raw:
        return None
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {i: str(v) for i, v in enumerate(raw)}


def _coerce_size(raw: Any) -> Optional[Tuple[int, int]]:
    if not raw:
        return None
    return (int(raw[0]), int(raw[1]))


def _coerce_norm(raw: Any) -> Optional[Tuple[float, ...]]:
    if not raw:
        return None
    return tuple(float(v) for v in raw)


def _artifact_key(p: Path) -> Tuple[int, ...]:
    """Cache key that changes whenever anything the description is derived from changes."""
    stamps = []
    for candidate in (p, p.parent / "model.pt", p.parent / "config.yaml"):
        try:
            stamps.append(candidate.stat().st_mtime_ns)
        except OSError:
            stamps.append(-1)
    return tuple(stamps)


def describe_artifact(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Facts for any artifact, `.pt` or exported graph.

    A `.engine` / `.onnx` / `.xml` carries no metadata of ours, but export writes it beside
    the `model.pt` it came from, so that checkpoint's meta (or the frozen sidecar config)
    describes it. Falls back to an empty description, letting the caller's explicit
    arguments stand.

    Cached on the artifact's mtime: reading six metadata fields costs a full deserialization
    of the weights, and `make export` describes the same run directory once per backend.
    Only the small info dict is retained - the state_dict it was read from is not.
    """
    p = Path(model_path)
    return deepcopy(_describe_artifact_cached(str(p), _artifact_key(p)))


@lru_cache(maxsize=16)
def _describe_artifact_cached(model_path: str, _key: Tuple[int, ...]) -> Dict[str, Any]:
    p = Path(model_path)
    if p.suffix == ".pt" and p.is_file():
        return load_and_describe(p)[1]

    sibling = p.parent / "model.pt"
    if sibling.is_file():
        # Via describe_artifact, not load_and_describe: every exported graph in a run dir
        # resolves to this one checkpoint, so they should share one read of it.
        return describe_artifact(sibling)

    cfg = sibling_config(p)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    names = _coerce_names(train_cfg.get("label_to_name"))
    model_name = cfg.get("model_name")
    mean = std = None
    if model_name:
        mean, std = resolve_norm(model_name)
    return {
        "model_name": model_name,
        "num_classes": len(names) if names else None,
        "label_to_name": names,
        "img_size": _coerce_size(train_cfg.get("img_size")),
        "mean": mean,
        "std": std,
    }
