from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray


def softmax(x: NDArray) -> NDArray:
    # axis=-1: a global reduction is only correct at batch 1
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class ONNXModel:
    def __init__(
        self,
        model_path: str,
        n_outputs: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,  # (h, w)
        half: bool = False,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        label_to_name: Optional[Dict[int, str]] = None,
    ):
        self.model_path = model_path
        self.label_to_name = label_to_name
        self.half = half
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.np_dtype = np.float16 if self.half else np.float32
        self.device = device if device else ("cuda" if ort.get_device() == "GPU" else "cpu")

        self._load_model()
        graph_size, graph_outputs, graph_batch = self._shapes_from_graph()
        self.input_size = tuple(input_size) if input_size is not None else graph_size
        self.n_outputs = n_outputs if n_outputs is not None else graph_outputs
        assert self.input_size, f"input size unknown for {model_path}; pass input_size="
        assert self.n_outputs, f"class count unknown for {model_path}; pass n_outputs="

        # The tighter of the graph's limit and the caller's cap; None when neither binds,
        # i.e. the batch axis is free and any N runs in one session call.
        limits = [n for n in (graph_batch, max_batch_size) if n is not None]
        self.max_batch_size: Optional[int] = min(limits) if limits else None
        self._test_pred()

    def _load_model(self):
        # ORT ships its CUDA kernels in a separate .so it dlopens at session creation. If
        # the CUDA/cuDNN it was built against is missing, that load fails and ORT quietly
        # runs on CPU instead - about 20x slower here, and easy to misread as a regression.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:  # best-effort; the guard below reports the real outcome
                logger.debug(f"ort.preload_dlls() failed: {e}")

        providers = ["CUDAExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        provider_options = (
            [{"cudnn_conv_algo_search": "HEURISTIC"}] if self.device == "cuda" else [{}]
        )
        self.model = ort.InferenceSession(
            self.model_path, providers=providers, provider_options=provider_options
        )

        active = self.model.get_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" not in active:
            logger.warning(
                "ONNX Runtime fell back to CPU: CUDAExecutionProvider was requested but is "
                f"not active (got {active}). Latency below is CPU latency, not a regression. "
                "onnxruntime-gpu must match the CUDA major version torch ships."
            )
            self.device = "cpu"
        logger.debug(f"ONNX Runtime providers: {active}")

    def _shapes_from_graph(self) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[int]]:
        """-> ((h, w), n_outputs, batch). A dynamic axis comes back as a string, not an
        int; for the batch axis that means no limit, reported as None."""
        shape = self.model.get_inputs()[0].shape
        size = None
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            size = (int(shape[2]), int(shape[3]))
        batch = int(shape[0]) if shape and isinstance(shape[0], int) else None
        out_shape = self.model.get_outputs()[0].shape
        outputs = int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else None
        return size, outputs, batch

    def _test_pred(self) -> None:
        self._predict(np.zeros((1, 3, *self.input_size), dtype=self.np_dtype))

    def _predict(self, inputs: NDArray) -> NDArray:
        # copy=False: _preprocess already emits np_dtype, a plain astype would copy the batch
        ort_inputs = {self.model.get_inputs()[0].name: inputs.astype(self.np_dtype, copy=False)}
        return self.model.run(None, ort_inputs)[0]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> normalized NCHW array. INTER_AREA matches training."""
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )  # cv2 takes (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        return ((img - mean) / std).astype(self.np_dtype)[None]

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
            # one image goes straight in: concatenating a single array still copies it
            logits = self._predict(chunk[0] if len(chunk) == 1 else np.concatenate(chunk))
            # (N, C) exactly: a graph that lost part of the batch raises instead of smearing rows
            rows.append(softmax(logits.astype(np.float32).reshape(len(chunk), self.n_outputs)))
        return np.concatenate(rows)

    def __call__(
        self, images: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> List[Dict[str, Union[int, float, str, Dict[str, float]]]]:
        """One {"label": name, "label_id": id, "score": its probability, "probs": full
        softmax by name} per image - top-1 is the label/score pair, everything past it is
        the caller's decision. A lone image gives a one-element list, so callers never
        branch on what they passed in. Names come from label_to_name when the wrapper knows
        it; classes without a name stringify their id."""
        probabilities = self.probs(images)
        names = getattr(self, "label_to_name", None) or {}
        out = []
        for row in probabilities:
            top = int(row.argmax())
            out.append(
                {
                    "label": str(names.get(top, top)),
                    "label_id": top,
                    "score": float(row[top]),
                    "probs": {str(names.get(j, j)): float(p) for j, p in enumerate(row)},
                }
            )
        return out
