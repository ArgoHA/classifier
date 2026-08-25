import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


def softmax(x: NDArray) -> NDArray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class CheckSideDetector:
    _instance = None

    MEAN_NORM = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD_NORM = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    INPUT_SIZE = (320, 320)
    CLASSES = ("front", "back", "other")
    N_OUTPUTS = len(CLASSES)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str):
        if hasattr(self, "model"):
            return

        self.model = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], provider_options=[{}]
        )
        self.input_name = self.model.get_inputs()[0].name
        self._test_pred()

    def _test_pred(self):
        input_blob = np.zeros((1, 3, *self.INPUT_SIZE), dtype=np.float32)
        _ = self._predict(input_blob)

    def _predict(self, inputs: NDArray) -> NDArray:
        ort_inputs = {self.input_name: inputs}
        outs = self.model.run(None, ort_inputs)
        return outs[0]

    def _preprocess(self, images: list[np.ndarray]) -> np.ndarray:
        imgs = np.empty((len(images), 3, *self.INPUT_SIZE), dtype=np.float32)
        for i, image in enumerate(images):
            img = cv2.resize(image, self.INPUT_SIZE[::-1], interpolation=cv2.INTER_AREA)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, then HWC to CHW
            img = np.ascontiguousarray(img, dtype=np.float32)
            imgs[i] = ((img / 255.0) - self.MEAN_NORM) / self.STD_NORM
        return imgs

    @staticmethod
    def _postprocess(logits: NDArray) -> list[dict]:
        probs = softmax(logits)
        return [
            {_cls: float(prob) for _cls, prob in zip(CheckSideDetector.CLASSES, prob)}
            for prob in probs
        ]

    def process(self, images: list[np.ndarray]) -> list[dict]:
        images = self._preprocess(images)
        logits = self._predict(images)
        return self._postprocess(logits)
