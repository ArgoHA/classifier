from typing import List, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch


class TensorRT_model:
    def __init__(
        self,
        model_path: str,
        n_outputs: int,
        input_size: Tuple[int, int] = (256, 256),  # (h, w)
        half: bool = False,
        device: str = None,
    ):
        self.mean_norm = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_norm = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = input_size
        self.n_outputs = n_outputs
        self.model_path = model_path
        self.half = half
        self.channels = 3

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.np_dtype = np.float16 if self.half else np.float32
        self.torch_dtype = torch.float16 if self.half else torch.float32

        self._load_engine()

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

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
        img = cv2.resize(
            image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA
        )  # (w, h)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # Convert to tensor and normalize on GPU
        tensor = torch.from_numpy(img).to(self.device, non_blocking=True)
        tensor = tensor.to(dtype=self.torch_dtype).div_(255.0)
        tensor = (
            tensor
            - torch.tensor(self.mean_norm, device=self.device, dtype=self.torch_dtype)[
                :, None, None
            ]
        ) / torch.tensor(self.std_norm, device=self.device, dtype=self.torch_dtype)[:, None, None]
        return tensor.unsqueeze(0).contiguous()  # Add batch dimension

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

    def _postprocess(self, outputs: torch.Tensor) -> Tuple[int, float]:
        out = outputs[0].squeeze().float().cpu().numpy()  # Shape becomes (n_outputs,)
        probs = np.exp(out) / (np.sum(np.exp(out)) + 1e-8)
        label = int(np.argmax(probs))
        max_prob = float(probs[label])
        return label, max_prob

    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        input_tensor = self._preprocess(image)
        outputs = self._predict(input_tensor)
        return self._postprocess(outputs)
