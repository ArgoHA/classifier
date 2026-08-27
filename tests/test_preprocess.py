"""The four wrappers each carry their own copy of the preprocessing, on purpose: every one
is meant to be liftable into a service as a single file. Duplication needs a drift guard,
and this is it - the four copies must produce a bit-identical tensor for the same image.

`_preprocess` only reads a handful of attributes, so the wrappers are built with
`__new__` and those attributes set directly. That keeps this test fast and GPU-free: no
engine, no session, no checkpoint.
"""

import numpy as np
import pytest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = (256, 192)  # deliberately non-square: catches an (h, w) / (w, h) swap


@pytest.fixture(scope="module")
def bgr_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)


def _reference(image, size=SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """What every wrapper must produce: INTER_AREA resize, BGR->RGB, CHW, /255, normalize."""
    import cv2

    img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    mean_a = np.asarray(mean, np.float32)[:, None, None]
    std_a = np.asarray(std, np.float32)[:, None, None]
    return ((img - mean_a) / std_a)[None]


def _bare(cls, **attrs):
    obj = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _numpy_wrapper(module_name, cls_name):
    module = pytest.importorskip(module_name)
    cls = getattr(module, cls_name)
    return _bare(
        cls,
        input_size=SIZE,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        np_dtype=np.float32,
        device="cpu",
    )


@pytest.mark.parametrize(
    "module_name,cls_name",
    [
        ("img_clf.infer.onnx_model", "ONNXModel"),
        ("img_clf.infer.ov_model", "OVModel"),
    ],
)
def test_numpy_wrappers_match_reference(bgr_image, module_name, cls_name):
    wrapper = _numpy_wrapper(module_name, cls_name)
    np.testing.assert_array_equal(wrapper._preprocess(bgr_image), _reference(bgr_image))


def test_torch_wrapper_matches_reference(bgr_image):
    module = pytest.importorskip("img_clf.infer.torch_model")
    wrapper = _numpy_wrapper("img_clf.infer.torch_model", "TorchModel")
    assert module  # keep the import meaningful for linters
    out = wrapper._preprocess(bgr_image).cpu().numpy()
    np.testing.assert_array_equal(out, _reference(bgr_image))


def test_tensorrt_wrapper_matches_reference(bgr_image):
    """TRT uploads uint8 and scales on the device, so it is the copy most likely to drift.
    Allowed a float32 epsilon: the ops are the same but the order is not."""
    pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("img_clf.infer.trt_model")

    wrapper = _bare(
        module.TRTModel,
        input_size=SIZE,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device="cpu",
        torch_dtype=torch.float32,
    )
    out = wrapper._preprocess(bgr_image).cpu().numpy()
    np.testing.assert_allclose(out, _reference(bgr_image), rtol=1e-6, atol=1e-6)


def test_non_square_size_is_not_transposed(bgr_image):
    """SIZE is (h, w) while cv2 takes (w, h) - the classic silent swap."""
    wrapper = _numpy_wrapper("img_clf.infer.onnx_model", "ONNXModel")
    assert wrapper._preprocess(bgr_image).shape == (1, 3, SIZE[0], SIZE[1])


def test_channel_order_is_rgb(bgr_image):
    """Channel 0 out must be the red channel in, i.e. BGR index 2."""
    wrapper = _numpy_wrapper("img_clf.infer.onnx_model", "ONNXModel")
    wrapper.input_size = bgr_image.shape[:2]  # no resize, so pixels map 1:1
    out = wrapper._preprocess(bgr_image)[0]
    expected = (bgr_image[:, :, 2].astype(np.float32) / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    np.testing.assert_allclose(out[0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "module_name,cls_name",
    [
        ("img_clf.infer.onnx_model", "ONNXModel"),
        ("img_clf.infer.ov_model", "OVModel"),
        ("img_clf.infer.trt_model", "TRTModel"),
        ("img_clf.infer.torch_model", "TorchModel"),
    ],
)
def test_wrappers_default_to_imagenet_stats(module_name, cls_name):
    """Checked on the constructor signature, not the module constant: a wrapper could
    define IMAGENET_MEAN and then not use it. The default is what a lifted file actually
    applies when a service constructs it with just a path."""
    import inspect

    module = pytest.importorskip(module_name)
    params = inspect.signature(getattr(module, cls_name).__init__).parameters
    assert tuple(params["mean"].default) == IMAGENET_MEAN, module_name
    assert tuple(params["std"].default) == IMAGENET_STD, module_name


def test_graph_wrappers_import_nothing_from_the_package():
    """The point of the duplication: `trt_model.py` / `onnx_model.py` / `ov_model.py` must
    be copyable into a service on their own. `torch_model.py` is exempt - a .pt needs the
    architecture and the checkpoint reader."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "img_clf" / "infer"
    pattern = re.compile(r"^\s*(?:from|import)\s+img_clf\b", re.MULTILINE)
    for name in ("trt_model.py", "onnx_model.py", "ov_model.py"):
        offenders = pattern.findall((root / name).read_text())
        assert not offenders, f"{name} imports from img_clf: {offenders}"


@pytest.mark.parametrize(
    "module_name",
    ["img_clf.infer.onnx_model", "img_clf.infer.ov_model", "img_clf.infer.trt_model"],
)
def test_softmax_copies_normalize_each_row(module_name):
    """Three package-free copies of one helper, each must normalize per row. TRT's copy
    once reduced over the whole batch, which is only right at batch 1."""
    module = pytest.importorskip(module_name)
    logits = np.random.default_rng(0).normal(size=(4, 7)).astype(np.float32) * 5
    out = module.softmax(logits)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    np.testing.assert_allclose(out, e / e.sum(axis=1, keepdims=True), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-6)
