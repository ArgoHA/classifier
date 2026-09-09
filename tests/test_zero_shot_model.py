"""ZeroShotModel over a fake open_clip dual encoder: shapes, order, cache, guards.

The hub calls are monkeypatched, so no downloads; a real checkpoint is one `make infer
ARGS="infer.zero_shot=true"` away.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from img_clf.infer.zero_shot_model import DEFAULT_TEMPLATE, ZeroShotModel


class FakeDualEncoder(torch.nn.Module):
    """Enough open_clip for the wrapper: logit_scale, encode_image/encode_text with
    normalize=, dtype-adaptive for half=True, fixed seed."""

    def __init__(self, dim: int = 8):
        super().__init__()
        torch.manual_seed(0)
        self.logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(50.0))))
        self.img_proj = torch.nn.Linear(3, dim)
        self.txt_proj = torch.nn.Linear(16, dim)
        self.text_calls = 0

    def encode_image(self, image, normalize: bool = False):
        feats = self.img_proj(image.to(self.img_proj.weight.dtype).mean(dim=(2, 3)))
        return feats / feats.norm(dim=-1, keepdim=True) if normalize else feats

    def encode_text(self, tokens, normalize: bool = False):
        self.text_calls += 1
        feats = self.txt_proj(tokens.to(self.txt_proj.weight.dtype))
        return feats / feats.norm(dim=-1, keepdim=True) if normalize else feats


def _fake_tokenize(texts):
    """16 int tokens per prompt, char codes - distinct labels get distinct rows."""
    width = 16
    rows = [[min(ord(c), 255) for c in text][:width] for text in texts]
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)


@pytest.fixture
def fake_open_clip(monkeypatch):
    """The wrapper does `import open_clip` inside _load_model - patch the module itself."""
    import open_clip

    cfg = {"size": 32, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "interpolation": "bicubic"}
    monkeypatch.setattr(
        open_clip,
        "create_model_and_transforms",
        lambda hub, device, precision, **_: (FakeDualEncoder(), None, None),
    )
    monkeypatch.setattr(open_clip, "get_tokenizer", lambda hub: _fake_tokenize)
    monkeypatch.setattr(open_clip, "get_model_preprocess_cfg", lambda model: cfg)


def _model(**kwargs) -> ZeroShotModel:
    kwargs.setdefault("hub", "hf-hub:timm/Fake-SigLIP2-32")
    kwargs.setdefault("labels", ["cat", "dog"])
    return ZeroShotModel(**kwargs)


@pytest.fixture
def model(fake_open_clip):
    return _model()


def _images(n, seed=0, size=(40, 60)):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (*size, 3), dtype=np.uint8) for _ in range(n)]


def test_probs_rows_sum_to_one(model):
    probs = model.probs(_images(4))
    assert probs.shape == (4, 2) and probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_single_image_is_a_one_row_batch(model):
    assert model.probs(_images(1)[0]).shape == (1, 2)


def test_batch_matches_per_image(model):
    images = _images(3, seed=2)
    rows = model.probs(images)
    for i, img in enumerate(images):
        np.testing.assert_allclose(model.probs([img])[0], rows[i], atol=1e-6)


def test_label_order_follows_the_requested_set(model):
    img = _images(1, seed=4)[0]
    probs = model.probs(img)
    flipped = model.probs(img, labels=["dog", "cat"])
    np.testing.assert_allclose(probs[0], flipped[0][::-1], atol=1e-6)


def test_labels_are_not_uniform(model):
    probs = model.probs(_images(1, seed=4))
    assert not np.allclose(probs[0], 0.5, atol=1e-3)


def test_call_matches_the_torchmodel_contract(model):
    img = _images(1)[0]
    preds = model(img)
    probs = model.probs(img)
    assert len(preds) == 1
    assert isinstance(preds[0]["label"], int)
    assert preds[0]["label"] == int(probs[0].argmax())
    assert preds[0]["prob"] == pytest.approx(float(probs[0].max()), abs=1e-6)


def test_mapping_labels_fix_the_ids(fake_open_clip):
    model = _model(labels={1: "cat", 0: "dog"})
    assert model.label_to_name == {0: "dog", 1: "cat"}
    assert model.label_names == ("dog", "cat")
    assert model.n_outputs == 2


def test_string_keys_in_a_mapping_are_normalized(fake_open_clip):
    model = _model(labels={"0": "cat", "1": "dog"})
    assert model.label_names == ("cat", "dog")


def test_text_is_encoded_at_construction_and_cached(model):
    img = _images(1)
    assert model.model.text_calls == 1  # _test_pred warmed the construction labels
    model.probs(img)
    model.probs(img)
    assert model.model.text_calls == 1
    model.probs(img, labels=["dog", "cat"])
    assert model.model.text_calls == 2
    model.probs(img)
    assert model.model.text_calls == 2  # the LRU still holds the construction set


def test_cache_is_bounded(fake_open_clip):
    model = _model(cache_size=2)
    img = _images(1)[0]
    for labels in (["a", "b"], ["c", "d"], ["e", "f"]):
        model.probs(img, labels=labels)
    assert model.model.text_calls == 4  # + the construction warm-up; "a"/"b" was evicted
    model.probs(img, labels=["a", "b"])
    assert model.model.text_calls == 5


def test_the_checkpoint_describes_the_preprocessing(model):
    assert model.input_size == (32, 32)
    assert model.mean == (0.5, 0.5, 0.5) and model.std == (0.5, 0.5, 0.5)
    assert model.n_outputs == 2
    assert model.max_batch_size is None  # like TorchModel: no graph-side limit


def test_half_keeps_the_output_contract(fake_open_clip):
    model = _model(half=True)
    probs = model.probs(_images(2))
    assert probs.dtype == np.float32 and probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_guards(model):
    with pytest.raises(ValueError, match="at least one image"):
        model.probs([])
    with pytest.raises(ValueError, match="at least two labels"):
        _model(labels=["only"])
    with pytest.raises(ValueError, match="unique"):
        _model(labels=["cat", "cat"])


def test_template_must_carry_the_label_slot():
    with pytest.raises(ValueError, match=r"\{label\}"):
        ZeroShotModel(hub="hf-hub:timm/Fake", labels=["a", "b"], template="no slot here")
    assert "{label}" in DEFAULT_TEMPLATE


def test_bare_model_name_is_rejected():
    """create_model_and_transforms would load 'ViT-B-32' with random weights."""
    with pytest.raises(ValueError, match="hf-hub:"):
        ZeroShotModel(hub="ViT-B-32", labels=["a", "b"])


def test_module_imports_nothing_from_the_package():
    """Same rule as the graph wrappers: liftable into a service as one file."""
    src = (
        Path(__file__).resolve().parents[1] / "img_clf" / "infer" / "zero_shot_model.py"
    ).read_text()
    assert not re.findall(r"^\s*(?:from|import)\s+img_clf\b", src, re.MULTILINE)
