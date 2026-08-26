"""The checkpoint envelope must not lose facts, and must keep reading old checkpoints."""

import torch

from img_clf.dl.ckpt import describe, save_checkpoint, unwrap_checkpoint


def _state_dict():
    return {"backbone.weight": torch.zeros(4, 3), "classifier.weight": torch.zeros(7, 4)}


def test_envelope_round_trip(tmp_path):
    meta = {
        "ckpt_version": 1,
        "model_name": "efficientnet_b0",
        "num_classes": 7,
        "img_size": [256, 256],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.25, 0.25, 0.25],
    }
    path = tmp_path / "model.pt"
    save_checkpoint(path, _state_dict(), meta)

    sd, read_meta = unwrap_checkpoint(torch.load(path, weights_only=True))
    assert set(sd) == set(_state_dict())
    assert read_meta == meta


def test_bare_state_dict_still_loads(tmp_path):
    """Checkpoints written before the envelope existed must keep working."""
    path = tmp_path / "legacy.pt"
    torch.save(_state_dict(), path)

    sd, meta = unwrap_checkpoint(torch.load(path, weights_only=True))
    assert set(sd) == set(_state_dict())
    assert meta == {}


def test_class_count_read_from_weights_not_meta():
    """The head's shape is ground truth; a stale meta must not override it."""
    info = describe(_state_dict(), {"num_classes": 999})
    assert info["num_classes"] == 7


def test_meta_only_fields_survive_describe():
    info = describe(_state_dict(), {"model_name": "resnet50", "img_size": [320, 224]})
    assert info["model_name"] == "resnet50"
    assert info["img_size"] == (320, 224)


def test_meta_must_be_plain_python(tmp_path):
    """A numpy scalar or OmegaConf node saves fine and then breaks every loader, because
    they all pass weights_only=True. Fail at the write instead."""
    import numpy as np
    import pytest

    with pytest.raises(TypeError, match="plain python"):
        save_checkpoint(tmp_path / "bad.pt", _state_dict(), {"best_f1": np.float32(0.9)})

    with pytest.raises(TypeError, match="plain python"):
        save_checkpoint(tmp_path / "bad2.pt", _state_dict(), {"sizes": [np.int64(256)]})


def test_valid_meta_round_trips_under_weights_only(tmp_path):
    path = tmp_path / "ok.pt"
    meta = {"model_name": "efficientnet_b0", "img_size": [256, 256], "mean": [0.5, 0.5, 0.5]}
    save_checkpoint(path, _state_dict(), meta)
    assert unwrap_checkpoint(torch.load(path, weights_only=True))[1] == meta
