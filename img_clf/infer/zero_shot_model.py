"""
Zero-shot classification with an open_clip dual-encoder checkpoint (SigLIP 2 / CLIP).
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

DEFAULT_TEMPLATE = "a photo of a {label}."

_INTERPOLATION = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}


def _label_to_name(labels: Union[Sequence[str], Mapping[int, str]]) -> Dict[int, str]:
    """Sequence or label_to_name mapping (DictConfig included) -> {id: name}, ids 0..N-1."""
    if hasattr(labels, "items"):
        items = {int(k): str(v) for k, v in labels.items()}
        if sorted(items) != list(range(len(items))):
            raise ValueError(f"label ids must be 0..N-1, got {sorted(items)}")
        names = [items[i] for i in range(len(items))]
    else:
        names = [str(name) for name in labels]
    if len(names) < 2:
        raise ValueError(f"zero-shot needs at least two labels, got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"labels must be unique, got {names}")
    return dict(enumerate(names))


class ZeroShotModel:
    def __init__(
        self,
        hub: str,
        labels: Union[Sequence[str], Mapping[int, str]],
        template: str = DEFAULT_TEMPLATE,
        half: bool = False,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        cache_size: int = 256,
    ):
        """`hub` is an 'hf-hub:' open_clip checkpoint reference (a bare name is rejected:
        create_model_and_transforms would load random weights). `labels` fixes the class
        set - a sequence or the config's label_to_name mapping, ids follow it."""

        if not hub.startswith("hf-hub:"):
            raise ValueError(
                f"hub must be an 'hf-hub:' reference, got {hub!r}: a bare name loads random "
                "weights without an explicit pretrained= tag"
            )
        if "{label}" not in template:
            raise ValueError(f"template must contain '{{label}}', got {template!r}")
        self.hub = hub
        self.template = template
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half
        # No graph-side limit, like TorchModel: the nn.Module takes any batch.
        self.max_batch_size: Optional[int] = max_batch_size
        self.label_to_name = _label_to_name(labels)
        self.label_names = tuple(self.label_to_name[i] for i in range(len(self.label_to_name)))
        self.n_outputs = len(self.label_names)
        self._cache_size = cache_size
        self._text_cache: "OrderedDict[Tuple[str, ...], torch.Tensor]" = OrderedDict()

        self._init_params()
        self._load_model()
        self._test_pred()

    def _init_params(self) -> None:
        self.np_dtype = np.float16 if self.half else np.float32

    def _load_model(self) -> None:
        """open_clip imports here, not at module scope: the file lifts into a finetune-only
        service, and the zero_shot extra is only needed when this wrapper is built."""
        import open_clip

        precision = "fp16" if self.half and self.device.startswith("cuda") else "fp32"
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.hub, device=self.device, precision=precision
        )
        if self.half and not self.device.startswith("cuda"):
            self.model.half()
        self.model.eval()
        if not hasattr(self.model, "logit_scale"):
            raise ValueError(f"{self.hub} has no logit_scale - not a zero-shot dual encoder")
        with torch.no_grad():
            self._logit_scale = self.model.logit_scale.exp().float()
        self._tokenizer = open_clip.get_tokenizer(self.hub)
        cfg = open_clip.get_model_preprocess_cfg(self.model)
        size = cfg.get("size", 224)
        self.input_size = (
            (int(size), int(size)) if isinstance(size, int) else (int(size[0]), int(size[1]))
        )
        self.mean = tuple(cfg.get("mean") or (0.5, 0.5, 0.5))
        self.std = tuple(cfg.get("std") or (0.5, 0.5, 0.5))
        self._interp = _INTERPOLATION.get(str(cfg.get("interpolation", "bicubic")), cv2.INTER_CUBIC)

    def _test_pred(self) -> None:
        """One forward through both towers: bad installs fail at construction; warms the
        text cache."""
        blank = np.zeros((*self.input_size, 3), dtype=np.uint8)
        self.probs(blank)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """BGR HWC uint8 -> normalized NCHW tensor; interpolation, size and stats from the
        checkpoint's own cfg, not the finetune convention."""
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=self._interp
        )  # cv2 takes (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        img = ((img - mean) / std).astype(self.np_dtype)[None]
        return torch.from_numpy(img).to(self.device)

    @torch.no_grad()
    def _text_features(self, names: Tuple[str, ...]) -> torch.Tensor:
        """Prompts -> normalized text embeddings, (L, D) float32, one row per label in
        order; cached per label tuple, LRU past `cache_size`."""
        cached = self._text_cache.get(names)
        if cached is not None:
            self._text_cache.move_to_end(names)
            return cached
        tokens = self._tokenizer([self.template.format(label=name) for name in names])
        feats = self.model.encode_text(tokens.to(self.device), normalize=True).float()
        self._text_cache[names] = feats
        while len(self._text_cache) > self._cache_size:
            self._text_cache.popitem(last=False)
        return feats

    @torch.no_grad()
    def probs(
        self,
        images: Union[np.ndarray, Sequence[np.ndarray]],
        labels: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
    ) -> np.ndarray:
        """Softmax rows, (N, C) float32, row i for images[i]; one BGR image counts as N=1.
        `labels=None` uses the construction set; a per-call set classifies against its own
        classes, ids following it. A sequence chunks to `max_batch_size`."""
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        if len(images) == 0:
            raise ValueError("probs() needs at least one image")
        names = self.label_names if labels is None else tuple(_label_to_name(labels).values())
        text = self._text_features(names)
        step = self.max_batch_size or len(images)
        rows = []
        for start in range(0, len(images), step):
            chunk = [self._preprocess(img) for img in images[start : start + step]]
            # one image goes straight in: torch.cat of a single tensor is still a copy kernel
            batch = chunk[0] if len(chunk) == 1 else torch.cat(chunk)
            image = self.model.encode_image(batch, normalize=True).float()
            logits = image @ text.T * self._logit_scale
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)

    def __call__(
        self, images: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> List[Dict[str, Union[int, float]]]:
        """One {"label": class id, "prob": its probability} per image - the TorchModel
        contract. Per-call label sets go through probs()."""
        probabilities = self.probs(images)
        return [
            {"label": int(label), "prob": float(probabilities[i, label])}
            for i, label in enumerate(probabilities.argmax(axis=1))
        ]
