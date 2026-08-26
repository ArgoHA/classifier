from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from openvino import Core


def softmax(x: NDArray) -> NDArray:
    # axis=-1: a global reduction is only correct at batch 1
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class OVModel:
    def __init__(
        self,
        model_path: str,
        n_outputs: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,  # (h, w)
        half: bool = False,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.model_path = model_path
        self.half = half
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.np_dtype = np.float16 if self.half else np.float32

        core = Core()
        graph = core.read_model(self.model_path)

        graph_size, graph_outputs, graph_batch = self._shapes_from_graph(graph)
        self.input_size = tuple(input_size) if input_size is not None else graph_size
        self.n_outputs = n_outputs if n_outputs is not None else graph_outputs
        assert self.input_size, f"input size unknown for {model_path}; pass input_size="
        assert self.n_outputs, f"class count unknown for {model_path}; pass n_outputs="

        # None = the batch axis is free on the compiled model, so any N runs in one call.
        self.max_batch_size: Optional[int] = self._load_model(core, graph, graph_batch)
        self._test_pred()

    @staticmethod
    def _shapes_from_graph(
        model,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[int]]:
        """-> ((h, w), n_outputs, batch) off the uncompiled graph; batch None when free."""
        size = outputs = None
        in_shape = model.inputs[0].partial_shape
        if len(in_shape) == 4 and in_shape[2].is_static and in_shape[3].is_static:
            size = (in_shape[2].get_length(), in_shape[3].get_length())
        batch = None if in_shape[0].is_dynamic else in_shape[0].get_length()
        out_shape = model.outputs[0].partial_shape
        if len(out_shape) and out_shape[-1].is_static:
            outputs = out_shape[-1].get_length()
        return size, outputs, batch

    def _load_model(self, core: Core, graph, graph_batch: Optional[int]) -> Optional[int]:
        """Compile `graph`; returns the batch limit the compiled model ended up with.

        CPU takes the graph as it is. Any other device gets an explicit shape: the batch
        axis stays free when the graph's is, and is pinned to 1 otherwise. Not every plugin
        can actually run a free batch axis - intel_gpu over a non-Intel OpenCL compiles it
        and then fails on the first infer - so a free axis is smoke-tested with a 2-image
        batch and, if that fails, recompiled at a fixed batch of 1 with batching emulated.
        That keeps a device that serves single images today serving them.
        """
        self.device_name = "GPU" if "GPU" in core.get_available_devices() else "CPU"
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16" if self.half else "f32",
        }
        if self.device_name == "CPU":
            self.model = core.compile_model(graph, self.device_name, config=config)
            return graph_batch

        batch = -1 if graph_batch is None else graph_batch
        graph.reshape({0: [batch, 3, *self.input_size]})
        self.model = core.compile_model(graph, self.device_name, config=config)
        if batch != -1:
            return graph_batch
        try:
            self._predict(np.zeros((2, 3, *self.input_size), dtype=self.np_dtype))
            return None
        except Exception as e:
            logger.warning(
                f"OpenVINO {self.device_name} compiled a free batch axis but cannot run it "
                f"({type(e).__name__}); recompiling at batch 1 and emulating batching"
            )
            logger.debug(str(e))
            graph.reshape({0: [1, 3, *self.input_size]})
            self.model = core.compile_model(graph, self.device_name, config=config)
            return 1

    def _test_pred(self) -> None:
        self._predict(np.zeros((1, 3, *self.input_size), dtype=self.np_dtype))

    def _predict(self, input_blob: NDArray) -> NDArray:
        return self.model(input_blob)[self.model.output(0)]

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
            rows.append(softmax(logits.astype(np.float32).reshape(len(chunk), -1)))
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
