import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import tensorrt as trt
import torch


def softmax(x: np.ndarray) -> np.ndarray:
    # axis=-1: a global reduction is only correct at batch 1
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class TRTModel:
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
        self.channels = 3
        self.mean = tuple(mean)
        self.std = tuple(std)

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.np_dtype = np.float16 if self.half else np.float32
        self.torch_dtype = torch.float16 if self.half else torch.float32

        self._load_engine()
        graph_size, graph_outputs = self._shapes_from_engine()
        self.input_size = tuple(input_size) if input_size is not None else graph_size
        self.n_outputs = n_outputs if n_outputs is not None else graph_outputs
        assert self.input_size, f"input size unknown for {model_path}; pass input_size="
        assert self.n_outputs, f"class count unknown for {model_path}; pass n_outputs="

        # The engine profile's max, or the caller's cap when that is tighter.
        engine_max = self._max_batch_from_engine()
        self.max_batch_size: int = (
            engine_max if max_batch_size is None else min(max_batch_size, engine_max)
        )

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def _shapes_from_engine(self) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """-> ((h, w), n_outputs) from the engine's own IO tensors.

        A dynamic axis reports -1, in which case that fact is not recoverable here and the
        caller has to say; hence the Nones.
        """
        size = outputs = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if len(shape) == 4 and shape[2] > 0 and shape[3] > 0:
                    size = (int(shape[2]), int(shape[3]))
            elif shape and shape[-1] > 0:
                outputs = int(shape[-1])
        return size, outputs

    def _max_batch_from_engine(self) -> int:
        """Largest batch the engine's optimization profile accepts.

        One query covers both kinds of engine: a dynamic-batch engine reports its profile's
        (min, opt, max), and a static engine reports its fixed shape three times over, so a
        batch-1 export comes back as 1.
        """
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                _, _, max_shape = self.engine.get_tensor_profile_shape(name, 0)
                return int(max_shape[0])
        raise RuntimeError(f"{self.model_path}: engine has no input tensor")

    @staticmethod
    def _torch_dtype_from_trt(trt_dtype):
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int8:
            return torch.int8
        else:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    @functools.cached_property
    def _norm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mean, std) as (3, 1, 1) device tensors, built once: rebuilding them per image
        cost ~30 us of the ~1.4 ms single-image latency on an RTX 5070 Ti."""
        mean = torch.as_tensor(self.mean, device=self.device, dtype=self.torch_dtype)
        std = torch.as_tensor(self.std, device=self.device, dtype=self.torch_dtype)
        return mean[:, None, None], std[:, None, None]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """BGR HWC uint8 -> normalized NCHW tensor, scaled on the device.

        Uploads uint8 and does the float math on the GPU: a third of the PCIe bytes of a
        float32 upload, and it measurably moves the single-image latency. INTER_AREA to
        match what training fed the model.
        """
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )  # cv2 takes (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device, non_blocking=True)
        tensor = tensor.to(dtype=self.torch_dtype).div_(255.0)
        mean, std = self._norm
        return ((tensor - mean) / std).unsqueeze(0).contiguous()

    def _predict(self, img: torch.Tensor) -> List[torch.Tensor]:
        batch_shape = tuple(img.shape)

        n_io = self.engine.num_io_tensors
        bindings: List[int] = [None] * n_io
        outputs: List[torch.Tensor] = []

        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dims = tuple(self.engine.get_tensor_shape(name))
            dt = self.engine.get_tensor_dtype(name)
            t_dt = self._torch_dtype_from_trt(dt)

            if mode == trt.TensorIOMode.INPUT:
                ok = self.context.set_input_shape(name, batch_shape)
                assert ok, f"Failed to set input shape for {name} -> {batch_shape}"
                bindings[i] = img.data_ptr()
            else:
                out_shape = (batch_shape[0],) + dims[1:]
                out = torch.empty(out_shape, dtype=t_dt, device=self.device)
                outputs.append(out)
                bindings[i] = out.data_ptr()

        self.context.execute_v2(bindings)
        return outputs

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
            logits = self._predict(chunk[0] if len(chunk) == 1 else torch.cat(chunk))[0]
            # (N, C) exactly: an engine that lost part of the batch raises instead of smearing rows
            rows.append(softmax(logits.float().cpu().numpy().reshape(len(chunk), self.n_outputs)))
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
