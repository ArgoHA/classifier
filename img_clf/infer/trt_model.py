from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch


class TRTModel:
    def __init__(
        self,
        model_path: str,
        n_outputs: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None,  # (h, w)
        half: bool = False,
        device: str = None,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.model_path = model_path
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
        mean = torch.as_tensor(self.mean, device=self.device, dtype=self.torch_dtype)[:, None, None]
        std = torch.as_tensor(self.std, device=self.device, dtype=self.torch_dtype)[:, None, None]
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

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        # max-subtracted: the raw form overflows on large logits
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def probs(self, image: np.ndarray) -> np.ndarray:
        """Full softmax vector.

        The export parity check compares these rather than the predicted label: a graph can
        shift every probability and still pick the same class, so the whole distribution
        shows drift before the argmax does.
        """
        logits = self._predict(self._preprocess(image))
        return self._softmax(logits[0].squeeze().float().cpu().numpy()).reshape(-1)

    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        """BGR image in, (label, probability of that label) out."""
        probabilities = self.probs(image)
        label = int(np.argmax(probabilities))
        return label, float(probabilities[label])
