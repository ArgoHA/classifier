"""Integration: an exported graph must agree with torch, and every wrapper must agree
with every other on the same image.

These need a trained run directory, so they skip rather than fail on a fresh clone.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_dir():
    """Latest experiment dir from config.yaml / config_bench.yaml, if one exists."""
    for name in ("config_bench.yaml", "config.yaml"):
        cfg_path = REPO_ROOT / name
        if not cfg_path.is_file():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        models_root = Path(str(cfg["train"]["root"])) / "output" / "models"
        if not models_root.is_dir():
            continue
        candidates = [d for d in sorted(models_root.iterdir()) if (d / "model.pt").is_file()]
        if candidates:
            return candidates[-1]
    return None


@pytest.fixture(scope="module")
def run_dir():
    d = _run_dir()
    if d is None:
        pytest.skip("no trained run directory available")
    return d


@pytest.fixture(scope="module")
def sample_image(run_dir):
    import cv2

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_path = Path(str(cfg["train"]["data_path"]))
    csv_path = data_path / "val.csv"
    if not csv_path.is_file():
        pytest.skip("no val.csv to read a real image from")
    import pandas as pd

    rel = pd.read_csv(csv_path, header=None).iloc[0, 0]
    img = cv2.imread(str(data_path / rel))
    if img is None:
        pytest.skip(f"could not read {rel}")
    return img


@pytest.mark.slow
def test_checkpoint_describes_itself(run_dir):
    from img_clf.dl.ckpt import inspect

    info = inspect(run_dir / "model.pt")
    assert info["model_name"]
    assert info["num_classes"] and info["num_classes"] > 1
    assert info["img_size"] and len(info["img_size"]) == 2
    assert info["mean"] and info["std"]


@pytest.mark.slow
@pytest.mark.gpu
def test_every_backend_matches_torch(run_dir, sample_image):
    """Cosine of the full softmax, not just argmax: drift shows up here first."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from img_clf.dl.ckpt import norm_kwargs
    from img_clf.infer.torch_model import Torch_model

    # The reference gets the checkpoint's normalization too: comparing two preprocessings
    # would blame the graph for a difference that is in the input.
    norms = norm_kwargs(run_dir)
    reference = Torch_model(model_path=str(run_dir / "model.pt"), **norms).probs(sample_image)

    built = []
    onnx_path = run_dir / "model.onnx"
    if onnx_path.is_file():
        from img_clf.infer.onnx_model import ONNX_model

        built.append(("ONNX", ONNX_model(model_path=str(onnx_path), **norms)))

    ov_path = run_dir / "model.xml"
    if ov_path.is_file():
        from img_clf.infer.ov_model import OV_model

        built.append(("OpenVINO", OV_model(model_path=str(ov_path), **norms)))

    trt_path = run_dir / "model.engine"
    if trt_path.is_file():
        from img_clf.infer.trt_model import TensorRT_model

        built.append(("TensorRT", TensorRT_model(model_path=str(trt_path), **norms)))

    if not built:
        pytest.skip("no exported artifacts to compare")

    for name, model in built:
        out = model.probs(sample_image)
        cosine = float(np.dot(reference, out) / (np.linalg.norm(reference) * np.linalg.norm(out)))
        assert cosine > 0.9999, f"{name} cosine {cosine}"
        assert int(np.argmax(out)) == int(np.argmax(reference)), f"{name} top-1 differs"


@pytest.mark.slow
@pytest.mark.gpu
def test_wrappers_need_only_a_path(run_dir):
    """The envelope's payoff: no model_name / n_outputs / input_size at the call site."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from img_clf.infer.torch_model import Torch_model

    model = Torch_model(model_path=str(run_dir / "model.pt"))
    assert model.model_name
    assert model.n_outputs > 1
    assert len(model.input_size) == 2
    assert len(model.mean) == 3 and len(model.std) == 3
