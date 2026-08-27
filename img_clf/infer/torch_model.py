from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import timm
import torch

from img_clf.dl.ckpt import describe_artifact, load_and_describe

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchModel:
    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        n_outputs: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,  # (h, w)
        half: bool = False,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        """Everything but `model_path` is optional: the checkpoint describes itself."""
        self.model_path = model_path
        self.model_name = self._fill_from_ckpt(
            model_path, n_outputs, input_size, mean, std, model_name
        )
        assert self.model_name, f"model_name unknown for {model_path}; pass model_name="
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = half
        # No graph-side limit: an nn.Module takes any batch, so a sequence runs as one
        # forward however many images it holds unless the caller caps it. Device memory is
        # otherwise the only bound.
        self.max_batch_size: Optional[int] = max_batch_size

        self._init_params()
        self._load_model()
        self._test_pred()

    def _fill_from_ckpt(self, model_path, n_outputs, input_size, mean, std, model_name=None):
        """Resolve whatever the caller left as None from the checkpoint's own meta."""
        needs = [v is None for v in (n_outputs, input_size, mean, std, model_name)]
        info = describe_artifact(model_path) if any(needs) else {}
        self.n_outputs = n_outputs if n_outputs is not None else info.get("num_classes")
        size = input_size if input_size is not None else info.get("img_size")
        self.input_size = tuple(size) if size is not None else None
        self.mean = tuple(mean) if mean is not None else info.get("mean") or IMAGENET_MEAN
        self.std = tuple(std) if std is not None else info.get("std") or IMAGENET_STD
        self.label_to_name = info.get("label_to_name")
        assert self.input_size, f"input size unknown for {model_path}; pass input_size="
        assert self.n_outputs, f"class count unknown for {model_path}; pass n_outputs="
        return info.get("model_name") if model_name is None else model_name

    def _init_params(self) -> None:
        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

    def _load_model(self):
        self.model = timm.create_model(
            self.model_name, pretrained=False, num_classes=self.n_outputs
        )
        checkpoint, _ = load_and_describe(self.model_path)
        self.model.load_state_dict(checkpoint)

        if self.half:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

    def _test_pred(self):
        img = np.zeros((3, *self.input_size), dtype=self.np_dtype)
        self.model(torch.from_numpy(img).to(self.device)[None])

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """BGR HWC uint8 -> normalized NCHW tensor. INTER_AREA matches training."""
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )  # cv2 takes (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        img = ((img - mean) / std).astype(self.np_dtype)[None]
        return torch.from_numpy(img).to(self.device)

    @torch.no_grad()
    def probs(self, images: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        """Softmax rows, (N, C) float32, row i for images[i]; one BGR image counts as N=1.
        Images may have any sizes, each is resized on its own. A sequence runs as batches of
        at most `max_batch_size`, so the caller never sees a profile/shape error."""
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        step = self.max_batch_size or len(images)
        rows = []
        for start in range(0, len(images), step):
            chunk = [self._preprocess(img) for img in images[start : start + step]]
            # one image goes straight in: torch.cat of a single tensor is still a copy kernel
            logits = self.model(chunk[0] if len(chunk) == 1 else torch.cat(chunk))
            rows.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(rows)

    def __call__(
        self, images: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> List[Dict[str, Union[int, float]]]:
        """One {"label": class id, "prob": its probability} per image. A lone image gives a
        one-element list, so callers never branch on what they passed in."""
        probabilities = self.probs(images)
        return [
            {"label": int(label), "prob": float(probabilities[i, label])}
            for i, label in enumerate(probabilities.argmax(axis=1))
        ]
