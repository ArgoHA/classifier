from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from openvino import Core


def softmax(x: NDArray) -> NDArray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class OVModel:
    def __init__(
        self,
        model_path: str,
        n_outputs: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,  # (h, w)
        half: bool = False,
        max_batch_size=1,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.model_path = model_path
        self.half = half
        self.max_batch_size = max_batch_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.np_dtype = np.float16 if self.half else np.float32

        core = Core()
        graph = core.read_model(self.model_path)

        graph_size, graph_outputs = self._shapes_from_graph(graph)
        self.input_size = tuple(input_size) if input_size is not None else graph_size
        self.n_outputs = n_outputs if n_outputs is not None else graph_outputs
        assert self.input_size, f"input size unknown for {model_path}; pass input_size="
        assert self.n_outputs, f"class count unknown for {model_path}; pass n_outputs="

        self._load_model(core, graph)
        self._test_pred()

    @staticmethod
    def _shapes_from_graph(model) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """-> ((h, w), n_outputs) off the uncompiled graph."""
        size = outputs = None
        in_shape = model.inputs[0].partial_shape
        if len(in_shape) == 4 and in_shape[2].is_static and in_shape[3].is_static:
            size = (in_shape[2].get_length(), in_shape[3].get_length())
        out_shape = model.outputs[0].partial_shape
        if len(out_shape) and out_shape[-1].is_static:
            outputs = out_shape[-1].get_length()
        return size, outputs

    def _load_model(self, core: Core, det_ov_model) -> None:
        self.device_name = "CPU"
        if "GPU" in core.get_available_devices():
            self.device_name = "GPU"
        if self.device_name != "CPU":
            det_ov_model.reshape({0: [1, 3, *self.input_size]})

        inference_hint = "f16" if self.half else "f32"
        inference_mode = "CUMULATIVE_THROUGHPUT" if self.max_batch_size > 1 else "LATENCY"
        self.model = core.compile_model(
            det_ov_model,
            self.device_name,
            config={"PERFORMANCE_HINT": inference_mode, "INFERENCE_PRECISION_HINT": inference_hint},
        )

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

    def probs(self, image: np.ndarray) -> np.ndarray:
        """Full softmax vector.

        The export parity check compares these rather than the predicted label: a graph can
        shift every probability and still pick the same class, so the whole distribution
        shows drift before the argmax does.
        """
        logits = self._predict(self._preprocess(image))
        return softmax(logits).reshape(-1)

    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        """BGR image in, (label, probability of that label) out."""
        probabilities = self.probs(image)
        label = int(np.argmax(probabilities))
        return label, float(probabilities[label])
