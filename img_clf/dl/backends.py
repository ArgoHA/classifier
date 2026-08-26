"""One registry of exported backends, so bench, export and the parity test agree.

`display` is the index column of `bench_metrics.csv` and the `format` column of
`parity.csv`. Those two reports used to be written from separate tables that had already
drifted - bench called OpenVINO "OV", export called it "OpenVINO" - so the two reports
about the same run could not be joined on backend.

This lives in `dl/`, not in `infer/`: it is tooling that walks a run directory, and the
wrappers under `img_clf/infer/` are deliberately standalone - one file is meant to be
liftable into a service on its own. Nothing here is imported by them. The backend runtime
is only imported when a backend is actually loaded.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional, Sequence, Union


@dataclass(frozen=True)
class Backend:
    display: str
    filename: str
    module: str
    cls_name: str

    def artifact(self, models_dir: Union[str, Path]) -> Path:
        return Path(models_dir) / self.filename

    def load(self, models_dir: Union[str, Path], **kwargs):
        """Instantiate the wrapper, or None if its artifact is not there.

        Lazy per backend: a broken openvino install must not hide an ONNX parity failure,
        so import and construction are the caller's to guard.
        """
        path = self.artifact(models_dir)
        if not path.exists():
            return None
        cls = getattr(import_module(self.module), self.cls_name)
        return cls(model_path=str(path), **kwargs)


# Order is the order bench reports rows in.
BACKENDS: Dict[str, Backend] = {
    "torch": Backend("torch", "model.pt", "img_clf.infer.torch_model", "Torch_model"),
    "tensorrt": Backend("TensorRT", "model.engine", "img_clf.infer.trt_model", "TensorRT_model"),
    "openvino": Backend("OpenVINO", "model.xml", "img_clf.infer.ov_model", "OV_model"),
    "onnx": Backend("ONNX", "model.onnx", "img_clf.infer.onnx_model", "ONNX_model"),
}

# torch is the source of an export, not one of its products. Order is the order they build
# in: OpenVINO and TensorRT both consume the ONNX graph.
EXPORT_FORMATS = ("onnx", "openvino", "tensorrt")


def resolve_formats(requested: Optional[Sequence[str]], known: Sequence[str], label: str) -> list:
    """null / missing -> every key in `known`; a list restricts it."""
    if not requested:
        return list(known)
    requested = [str(f).lower() for f in requested]
    unknown = set(requested) - set(known)
    if unknown:
        raise ValueError(f"unknown {label} {sorted(unknown)}; pick from {list(known)}")
    return requested


def norm_kwargs(models_dir: Union[str, Path]) -> Dict[str, Sequence[float]]:
    """Trained normalization for the wrappers, from the checkpoint beside the graphs.

    The graph wrappers read shape and class count off their own graph, but normalization is
    a property of the training run rather than of the graph, so they can only default to
    ImageNet stats. Anything driving them from a run directory should pass the real values;
    a model trained with, say, inception stats is otherwise silently mis-normalized.
    """
    from img_clf.dl.ckpt import describe_artifact

    info = describe_artifact(Path(models_dir) / "model.pt")
    if info.get("mean") and info.get("std"):
        return {"mean": info["mean"], "std": info["std"]}
    return {}
