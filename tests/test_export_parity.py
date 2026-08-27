"""Integration: an exported graph must agree with torch, and every wrapper must agree
with every other on the same image - singly and as a batch.

These need a trained run directory, so they skip rather than fail on a fresh clone.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
N_SAMPLE = 8  # real val images, spread over the split so several classes are represented
COSINE_MIN = 0.9999
# Batched and single-image rows of one backend may come off different kernels: cuDNN's
# default TF32 convolutions put them up to ~3.5e-3 apart on efficientnet_b0 probabilities
# (TensorRT happens to be exact). 1e-2 leaves margin and still catches a swapped row or a
# softmax over the wrong axis, which differ by ~1. The top-1 must not move at all.
BATCH_ATOL = 1e-2


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


def _needs_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


@pytest.fixture(scope="module")
def run_dir():
    d = _run_dir()
    if d is None:
        pytest.skip("no trained run directory available")
    return d


@pytest.fixture(scope="module")
def sample_images(run_dir):
    import cv2
    import pandas as pd

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_path = Path(str(cfg["train"]["data_path"]))
    csv_path = data_path / "val.csv"
    if not csv_path.is_file():
        pytest.skip("no val.csv to read real images from")

    split = pd.read_csv(csv_path, header=None)
    step = max(1, len(split) // N_SAMPLE)
    images = []
    for idx in range(0, len(split), step):
        if len(images) >= N_SAMPLE:
            break
        img = cv2.imread(str(data_path / split.iloc[idx, 0]))
        if img is not None:
            images.append(img)
    if not images:
        pytest.skip("could not read any val image")
    return images


@pytest.fixture(scope="module")
def reference(run_dir):
    """torch, with the checkpoint's own normalization: comparing two preprocessings would
    blame the graph for a difference that is in the input."""
    _needs_cuda()
    from img_clf.dl.ckpt import norm_kwargs
    from img_clf.infer.torch_model import TorchModel

    return TorchModel(model_path=str(run_dir / "model.pt"), **norm_kwargs(run_dir))


@pytest.fixture(scope="module")
def backends(run_dir):
    """(name, wrapper) for every exported graph present in the run dir, built once."""
    _needs_cuda()
    from img_clf.dl.ckpt import norm_kwargs

    norms = norm_kwargs(run_dir)
    built = []
    onnx_path = run_dir / "model.onnx"
    if onnx_path.is_file():
        from img_clf.infer.onnx_model import ONNXModel

        built.append(("ONNX", ONNXModel(model_path=str(onnx_path), **norms)))

    ov_path = run_dir / "model.xml"
    if ov_path.is_file():
        from img_clf.infer.ov_model import OVModel

        built.append(("OpenVINO", OVModel(model_path=str(ov_path), **norms)))

    trt_path = run_dir / "model.engine"
    if trt_path.is_file():
        from img_clf.infer.trt_model import TRTModel

        built.append(("TensorRT", TRTModel(model_path=str(trt_path), **norms)))

    if not built:
        pytest.skip("no exported artifacts to compare")
    return built


def _cosine(a, b) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _assert_rows_close(name: str, batch: np.ndarray, single: np.ndarray) -> None:
    """The batched-path contract: same answers, same distributions up to kernel noise."""
    assert batch.shape == single.shape, name
    assert batch.dtype == np.float32, name
    np.testing.assert_array_equal(
        batch.argmax(1), single.argmax(1), err_msg=f"{name}: batched top-1 differs"
    )
    np.testing.assert_allclose(batch, single, atol=BATCH_ATOL, rtol=0, err_msg=name)


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
def test_every_backend_matches_torch(reference, backends, sample_images):
    """Cosine of the full softmax, not just argmax: drift shows up here first."""
    for img in sample_images:
        ref = reference.probs(img)[0]
        for name, model in backends:
            out = model.probs(img)[0]
            assert _cosine(ref, out) > COSINE_MIN, f"{name} cosine {_cosine(ref, out)}"
            assert int(np.argmax(out)) == int(np.argmax(ref)), f"{name} top-1 differs"


@pytest.mark.slow
@pytest.mark.gpu
def test_batch_matches_torch_batch(reference, backends, sample_images):
    """One probs call over the sample per backend against one torch call - the batched twin
    of the check above: row-wise cosine, then top-1, shape and dtype."""
    ref = reference.probs(sample_images)
    assert ref.shape == (len(sample_images), reference.n_outputs)
    for name, model in backends:
        out = model.probs(sample_images)
        assert out.shape == ref.shape and out.dtype == np.float32, name
        for i, (r, o) in enumerate(zip(ref, out)):
            assert _cosine(r, o) > COSINE_MIN, f"{name} row {i} cosine {_cosine(r, o)}"
        np.testing.assert_array_equal(
            out.argmax(1), ref.argmax(1), err_msg=f"{name}: batch top-1 differs from torch"
        )


@pytest.mark.slow
@pytest.mark.gpu
def test_batch_matches_single(reference, backends, sample_images):
    """A batched forward must not change any answer - torch included."""
    for name, model in [("torch", reference)] + backends:
        single = np.concatenate([model.probs(img) for img in sample_images])
        _assert_rows_close(name, model.probs(sample_images), single)


@pytest.mark.slow
@pytest.mark.gpu
def test_chunking_preserves_order(reference, backends, sample_images):
    """N = max_batch_size + 3 spills into a second chunk; rows must come back in input
    order and equal to the per-image results. TensorRT has a real profile limit (1 for a
    static engine, which makes this the emulated path); torch has none, so it gets a small
    one monkeypatched in."""
    cases = [("torch", reference, 3)]
    cases += [(name, m, m.max_batch_size) for name, m in backends if name == "TensorRT"]
    for name, model, limit in cases:
        n = limit + 3
        images = [sample_images[i % len(sample_images)] for i in range(n)]
        single = np.concatenate([model.probs(img) for img in images])
        saved = model.max_batch_size
        model.max_batch_size = limit
        try:
            out = model.probs(images)
        finally:
            model.max_batch_size = saved
        assert out.shape[0] == n, name
        _assert_rows_close(f"{name} chunked at {limit}", out, single)


@pytest.mark.slow
@pytest.mark.gpu
def test_single_image_batch_is_probs(reference, backends, sample_images):
    """One image and a list of that one image take the same path, so this one is exact -
    for probs and for __call__, which always answers with one dict per image."""
    img = sample_images[0]
    for name, model in [("torch", reference)] + backends:
        np.testing.assert_array_equal(model.probs([img]), model.probs(img), err_msg=name)
        prediction = model(img)
        assert model([img]) == prediction, name
        assert len(prediction) == 1 and set(prediction[0]) == {"label", "prob"}, name
        assert isinstance(prediction[0]["label"], int), name
        assert isinstance(prediction[0]["prob"], float), name


@pytest.mark.slow
@pytest.mark.gpu
def test_empty_batch_raises(reference, backends):
    for name, model in [("torch", reference)] + backends:
        with pytest.raises(ValueError):
            model.probs([])


@pytest.mark.slow
@pytest.mark.gpu
def test_mixed_sizes_in_one_batch(reference, backends, sample_images):
    """Each image is resized on its own, so a batch of different-sized inputs must equal
    the per-image results for those same inputs - and __call__ on the batch must agree."""
    import cv2

    base = sample_images[0]
    images = [base, cv2.resize(base, (97, 211)), cv2.resize(base, (640, 480)), base[20:, :150]]
    for name, model in [("torch", reference)] + backends:
        single = np.concatenate([model.probs(img) for img in images])
        _assert_rows_close(name, model.probs(images), single)
        labels = [pred["label"] for pred in model(images)]
        assert labels == single.argmax(1).tolist(), name


@pytest.mark.slow
@pytest.mark.gpu
def test_max_batch_size_is_reported(reference, backends):
    """The attribute the serving node chunks by: None for no graph-side limit, else a
    positive int. An engine always has a profile, so TensorRT is never None."""
    assert reference.max_batch_size is None
    for name, model in backends:
        limit = model.max_batch_size
        assert limit is None or (isinstance(limit, int) and limit >= 1), (name, limit)
    trt = dict(backends).get("TensorRT")
    if trt is not None:
        assert isinstance(trt.max_batch_size, int)


@pytest.mark.slow
@pytest.mark.gpu
def test_max_batch_size_kwarg_caps(reference, backends):
    """The D-FINE-seg-style constructor cap. It never rises above the graph's own limit, and
    a cap of 1 keeps any export on the per-image path - what a node still passing the old
    default gets."""
    for name, model in [("torch", reference)] + backends:
        capped = type(model)(
            model_path=model.model_path, max_batch_size=1, mean=model.mean, std=model.std
        )
        assert capped.max_batch_size == 1, name


@pytest.mark.slow
@pytest.mark.gpu
def test_wrappers_need_only_a_path(run_dir):
    """The envelope's payoff: no model_name / n_outputs / input_size at the call site."""
    _needs_cuda()
    from img_clf.infer.torch_model import TorchModel

    model = TorchModel(model_path=str(run_dir / "model.pt"))
    assert model.model_name
    assert model.n_outputs > 1
    assert len(model.input_size) == 2
    assert len(model.mean) == 3 and len(model.std) == 3
