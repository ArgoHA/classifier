"""Averaging mode must follow the configured class count, not the labels in the batch."""

import numpy as np

from img_clf.dl.validator import Validator


def _names(n):
    return {i: f"class_{i}" for i in range(n)}


def test_average_follows_configured_class_count():
    assert Validator(10, _names(10)).average == "macro"
    assert Validator(2, _names(2)).average == "binary"


def test_split_missing_classes_keeps_macro_averaging():
    """Bug #3: only two of ten classes present used to flip macro -> binary silently."""
    validator = Validator(10, _names(10))
    gt = [0, 1, 0, 1]
    preds = [0, 1, 1, 1]

    assert validator.average == "macro"
    metrics, per_class = validator.get_metrics(gt, preds, per_class=True)
    # Macro over all 10 configured classes: the 8 unseen ones count as 0, so this is well
    # below the 0.8 that a silent switch to binary averaging would have reported.
    assert metrics["f1"] < 0.2
    assert len(per_class) == 10


def test_macro_average_is_pinned_to_configured_labels():
    """Same predictions, two configured class counts -> different, well-defined macros."""
    gt, preds = [0, 1, 0, 1], [0, 1, 1, 1]
    two = Validator(2, _names(2)).get_metrics(gt, preds)[0]["f1"]
    ten = Validator(10, _names(10)).get_metrics(gt, preds)[0]["f1"]
    assert two > ten  # 10-class macro divides the same score over 10 slots


def test_per_class_metrics_cover_every_configured_class():
    validator = Validator(3, _names(3))
    _, per_class = validator.get_metrics([0, 0, 1], [0, 1, 1], per_class=True)
    assert set(per_class) == {"class_0", "class_1", "class_2"}


def test_threshold_sweep_picks_f1_optimum(tmp_path):
    validator = Validator(2, {0: "neg", 1: "pos"})
    # positive-class scores: a 0.5 cut misses the 0.35 positive, a lower cut catches it
    probs = np.array([[0.9, 0.1], [0.65, 0.35], [0.2, 0.8], [0.95, 0.05]])
    best = validator.threshold_sweep(probs, [0, 1, 1, 0], tmp_path, "val")

    assert best is not None
    assert best["best_threshold"] <= 0.35
    assert best["best_f1"] == 1.0
    assert (tmp_path / "val_threshold_sweep.png").is_file()


def test_threshold_sweep_is_binary_only(tmp_path):
    validator = Validator(3, _names(3))
    assert validator.threshold_sweep(np.ones((4, 3)) / 3, [0, 1, 2, 0], tmp_path, "val") is None


def test_confusion_matrix_shape_and_counts(tmp_path):
    validator = Validator(3, _names(3))
    matrix = validator.save_confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], tmp_path, "test")
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert (tmp_path / "test_confusion_matrix.png").is_file()
