from typing import Dict, Tuple

import cv2
import numpy as np
import timm
import torch


class Torch_model:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        n_outputs: int,
        input_size: Tuple[int, int] = (256, 256),  # (h, w)
        half: bool = False,
    ):
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.model_name = model_name
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = half

        self._init_params()
        self._load_model()
        self._test_pred()

    def _init_params(self) -> None:
        if self.half:
            self.np_dtype = np.float16
        else:
            self.np_dtype = np.float32

    def _load_model(self):
        self.model = timm.create_model(
            self.model_name, pretrained=False, num_classes=self.n_outputs
        )
        checkpoint = torch.load(self.model_path, weights_only=True, map_location="cpu")
        self.model.load_state_dict(checkpoint)

        if self.half:
            self.model.half()
        self.model.eval()
        self.model.to(self.device)

    def _test_pred(self):
        img = np.zeros((3, *self.input_size), dtype=self.np_dtype)
        self.model(torch.from_numpy(img).to(self.device)[None])

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )  # (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=self.np_dtype)
        img = (img / 255.0).astype(self.np_dtype)
        img = (img - self.mean_norm[:, None, None]) / self.std_norm[:, None, None]
        img = img[None]  # batch dim
        img = torch.from_numpy(img)
        img = img.half() if self.half else img
        return img.to(self.device)

    def _postprocess(self, logits: torch.Tensor) -> Tuple[int, float]:
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        label = int(np.argmax(probs))
        return label, np.max(probs)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        """
        cv2 image as input, return label, max_prob
        """
        image = self._preprocess(image)
        logits = self.model(image)
        label, max_prob = self._postprocess(logits)
        return label, max_prob
